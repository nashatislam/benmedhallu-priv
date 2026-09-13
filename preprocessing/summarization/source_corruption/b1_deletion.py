# -*- coding: utf-8 -*-
"""
Pipeline 1: deletion. The Leave-N-Out operation.

Adapted from Suhas BN et al., "Fact-Controlled Diagnosis of Hallucinations in
Medical Text Summarization", Interspeech 2025, which introduced the method on
English clinical dialogue (ACI-Bench). The method is theirs; the adaptation to
Bangla consumer health questions is ours. Cite them in the methodology, not in
related work.

Remove N atomic facts from the source. The untouched summary still asserts them,
so its claims become UNSUPPORTED - the source has fallen silent on them.

    Stage 1  common.EXTRACT_FACTS_PROMPT   summary          -> atomic facts
    Stage 2  build_jobs                    facts, n_values  -> one job per (doc, N)
    Stage 3  REMOVE_FACTS_PROMPT           source, targets  -> rewritten source

N is the difficulty dial and the whole point of the method. Running 1, 2 and 3
lets you plot detector accuracy against it rather than binning items into
difficulty tiers by hand.
"""

import random

from common import (  # noqa: F401  - re-exported so a notebook imports one module
    EXTRACT_FACTS_PROMPT,
    FACT_CATEGORIES,
    format_target,
    format_targets,
    parse_facts,
    parse_rewritten_source,
    parse_spans,
    targetable,
    DELETION_EXCLUDED,
)


# ---------------------------------------------------------------------------
# Stage 3 - the corruption step
# ---------------------------------------------------------------------------

REMOVE_FACTS_PROMPT = """You are given a Bangla medical source document written by a patient, and a list of facts to remove from it.

Your task is to rewrite the source document so that it no longer contains or supports any of the listed facts, while keeping everything else intact.

Rules:
1. Remove EVERY mention of each listed fact. A patient often repeats the same detail several times in different words, so search the whole document, not just the first occurrence.
2. Remove indirect references, paraphrases, pronouns, and follow-on clauses that refer to a listed fact.
3. After rewriting, the listed facts must not be inferable from what remains. If a reader could still work the fact out from the surrounding context, you have not removed enough.
4. Do NOT signal the removal. Do not write that the information is unavailable, unknown, or not mentioned. The rewritten document must read as if the fact was never part of the patient's account.
5. Do NOT replace a removed fact with a different fact or value. This is removal, not substitution. The document must stop speaking about the detail rather than say something else about it.
6. Do NOT remove, alter, shorten, or reword any information that is not on the list. Every other detail must survive with its original meaning.
7. Do NOT add any new information, clarification, or clinical detail.
8. Keep the patient's original register: informal Bangla conversational speech, including the original spelling style, punctuation habits, and level of fluency. Do not make the text more formal or more grammatical than the original.
9. Keep the remaining sentences in their original order.
10. Preserve the questions the patient asks at the end, unless a question itself states a listed fact.
11. The rewritten document must be coherent and natural to a native Bangla speaker reading it as a patient's message.
12. If a listed fact cannot be removed without making the document incoherent or destroying its main complaint, do not attempt a partial removal. Return the failure token instead.
13. Do not provide explanations, notes, or a description of what you changed.
14. Return your answer in EXACTLY the following format:

Original Span: <the exact text you deleted from the source; if several, separate with " ||| ">
New Span: NONE

Rewritten Source:
<the full rewritten Bangla document>

or, if rule 12 applies:

Rewritten Source:
REMOVAL_FAILED

Example:

Source document:
আমার ১০.০৯.১৯ তারিখ থেকে ডেঙ্গু জ্বর হইতেছে। আজকে ১০ দিন হল। এনএস১ পজিটিভ আসছে। এর পর থেকে প্লাটিলেট কমে যাচ্ছিল। গত ৩ দিন যাবত প্লাটিলেট বাড়ছে। সর্বশেষ আজকের রিপোর্টে ১৫০০০০ আসছে। আমার শরীর কিছুটা দুর্বল। এছাড়া আর তেমন কোন সমস্যা নেই। আমি কখন বুঝতে পারব যে আমার ডেঙ্গু জ্বর ভাল হয়ে গেছে? আমার কি আর সিবিসি টেস্ট করার দরকার আছে?

Facts to remove:
1. [Test Result] আজকের রিপোর্টে প্লাটিলেট ১৫০০০০ আসছে।

Output:

Original Span: সর্বশেষ আজকের রিপোর্টে ১৫০০০০ আসছে।
New Span: NONE

Rewritten Source:
আমার ১০.০৯.১৯ তারিখ থেকে ডেঙ্গু জ্বর হইতেছে। আজকে ১০ দিন হল। এনএস১ পজিটিভ আসছে। এর পর থেকে প্লাটিলেট কমে যাচ্ছিল। গত ৩ দিন যাবত প্লাটিলেট বাড়ছে। আমার শরীর কিছুটা দুর্বল। এছাড়া আর তেমন কোন সমস্যা নেই। আমি কখন বুঝতে পারব যে আমার ডেঙ্গু জ্বর ভাল হয়ে গেছে? আমার কি আর সিবিসি টেস্ট করার দরকার আছে?

Now perform the removal.

Source document:
{source}

Facts to remove:
{facts_to_remove}
"""


# ---------------------------------------------------------------------------
# Stage 2 - job enumeration
# ---------------------------------------------------------------------------

def build_jobs(doc_id, source, facts, n_values=(1, 2, 3), seed=0):
    """Enumerate one generation job per value of N, for one document.

    A job is one API call. Returns a list of dicts ready to be rendered into
    REMOVE_FACTS_PROMPT and written to the output CSV.

    Targets at N=2 are a superset of the N=1 target for the same document, so
    a change in detector accuracy is attributable to the facts that were added
    rather than to a different draw.

    At least one fact is always left standing, or the summary carries no signal.

    Deletion keeps "Other" as a target category, unlike alteration: it only
    needs a removable span, and Other facts have spans even when they have no
    swappable value.
    """
    rng = random.Random("{}|{}".format(seed, doc_id))
    pool = targetable(facts, doc_id, DELETION_EXCLUDED, seed)
    rng.shuffle(pool)

    jobs = []
    for n in n_values:
        if len(pool) < n or len(facts) < n + 1:
            continue
        targets = pool[:n]
        jobs.append(
            {
                "doc_id": doc_id,
                "pipeline": "deletion",
                "operation": "delete",
                "n": n,
                "targets": targets,
                "target_categories": [category for _, category, _ in targets],
                "retained": [f for f in facts if f not in targets],
                "prompt": REMOVE_FACTS_PROMPT.format(
                    source=source, facts_to_remove=format_targets(targets)
                ),
            }
        )
    return jobs
