# -*- coding: utf-8 -*-
"""
Pipeline 2: alteration. Two methods, one API call each.

Where deletion makes the summary's claim UNSUPPORTED, alteration makes it
CONTRADICTED: the source still discusses the detail and now says something the
summary cannot be reconciled with. That is the intrinsic versus extrinsic split
realised on the source side, and detectors behave very differently across it.

    Stage 1  common.EXTRACT_FACTS_PROMPT   summary            -> atomic facts
    Stage 2  build_jobs                    facts, methods     -> one job per (doc, method)
    Stage 3  build_prompt                  source, fact, meth -> rewritten source

This pipeline holds N at 1 and varies the METHOD; the deletion pipeline holds
the operation fixed and varies N. Exactly one axis moves in each, so a drop in
detector accuracy is always attributable.

Why N=1 here. Deletion compounds cleanly - removing three facts is just a harder
version of removing one. Altering three facts at once does not compound, it
produces a document rewritten in three unrelated ways, and a detector failure
cannot be attributed to any of them. Keep alteration to one fact per item.

The two methods:

    SUB   Substitute   swap the value for a plausible neighbour        borrowed
    COA   Coarsen      replace a specific value with a vague one       novel

SUB is the cited comparison arm: it is the operation FactCC and Faithfulness-QA
already use, and it reuses the NER track's validated entity-swap machinery. COA
carries the novelty - it is the only operation producing partial support, where
the direction of travel stays supported while the specific value does not.

Weaken, Reattribute and Conflict were built in the 50-document pilot and cut on
evidence. The measurement was claim survival: how much of what the source
originally said is still present verbatim after the edit.

    Delete 21%   Substitute 60%   Coarsen 65%  <- remove support
    Reattribute 96%   Weaken 97%   Conflict 100%  <- leave the claim findable

An operation that removes nothing leaves the summary's claim verbatim in the
source, which makes the gold label arguable. Reattribute failed twice over:
only 60% of its items introduced a person at all, and just two of fifty
summaries mention a third party, so every reattribution was an invention that
collided with a first-person complaint.
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
    ALTERATION_EXCLUDED,
)


# The seven categories kept after the pilot. Medical Procedure was dropped as
# half-mislabelled and tiny (5 facts / 50 docs), Specialist because patients do
# not name specialties in these summaries (1 fact / 50 docs), and Other because
# it bundled treatment responses, patient actions and stray measurements.
ALTERABLE_CATEGORIES = [
    "Symptom",
    "Health Condition",
    "Medicine",
    "Age",
    "Dosage",
    "Test Result",
    "Temporal",
]

# Coarsen works where a vaguer form is natural. Age is excluded: "my age is
# fairly low" is not something a patient writes, even though it passes the
# mechanical check.
COARSENABLE_CATEGORIES = [
    "Temporal",
    "Test Result",
    "Dosage",
]


# ---------------------------------------------------------------------------
# Stage 3 - shared skeleton, one instruction block per method
# ---------------------------------------------------------------------------

_SKELETON = """You are given a Bangla medical source document written by a patient, and one fact drawn from its summary.

{task}

Rules:
{method_rules}
{shared_rules}

Example:

Source document:
{ex_source}

Target fact:
{ex_fact}

Output:

Original Span: {ex_original}
New Span: {ex_new}

Rewritten Source:
{ex_rewritten}

Now perform the edit.

Source document:
{{source}}

Target fact:
{{target_fact}}
"""

# Rules every method obeys, numbered from 6 so each method gets 1-5 of its own.
_SHARED_RULES = """6. Change EVERY mention of the target detail in the same way. A patient often repeats a detail in different words across the message, and if some mentions carry the old version and others the new one, the document contradicts itself. That is a different condition from the one this task produces.
7. Do NOT change, remove, shorten, or reword any information other than the target detail. Every other fact must survive with its original meaning.
8. Do NOT add any information beyond what the edit itself requires.
9. Do NOT signal the change. Do not write that anything was corrected, updated, clarified or revised.
10. Keep the patient's original register: informal Bangla conversational speech, including the original spelling style, punctuation habits, and level of fluency. Do not make the text more formal or more grammatical than the original.
11. Keep the remaining sentences in their original order.
12. Preserve the questions the patient asks at the end, unless a question itself states the target fact, in which case edit it consistently with rule 6.
13. The rewritten document must read naturally to a native Bangla speaker as a patient's message.
14. If the edit cannot be made without destroying the document's main complaint, do not attempt a partial edit. Put REWRITE_FAILED on the Rewritten Source line instead.
15. Do not provide explanations, notes, or a description of what you changed.
16. Return your answer in EXACTLY the format shown in the example: an "Original Span:" line, a "New Span:" line, then "Rewritten Source:" followed by the full document."""


_DENGUE = ("আমার ১০.০৯.১৯ তারিখ থেকে ডেঙ্গু জ্বর হইতেছে। আজকে ১০ দিন হল। এনএস১ পজিটিভ আসছে। "
           "এর পর থেকে প্লাটিলেট কমে যাচ্ছিল। গত ৩ দিন যাবত প্লাটিলেট বাড়ছে। "
           "সর্বশেষ আজকের রিপোর্টে ১৫০০০০ আসছে। আমার শরীর কিছুটা দুর্বল। "
           "এছাড়া আর তেমন কোন সমস্যা নেই। আমি কখন বুঝতে পারব যে আমার ডেঙ্গু জ্বর ভাল হয়ে গেছে? "
           "আমার কি আর সিবিসি টেস্ট করার দরকার আছে?")


def _dengue_with(old, new):
    return _DENGUE.replace(old, new)


METHODS = {
    "SUB": {
        "name": "Substitute",
        "task": "Your task is to rewrite the source document so that it asserts a DIFFERENT value for the detail named in the target fact, while keeping everything else intact.",
        "rules": """1. Find where the document states the target detail and change what it says, so the document now asserts a different value for that same detail.
2. The new value must CONTRADICT the original, not merely reword it. After the rewrite, the target fact and the document must not both be able to be true.
3. The difference must be large enough to matter clinically. Do not make a token change a reader would treat as the same fact.
4. Replace like with like, according to the category in brackets before the target fact:
   [Health Condition] another condition that could be confused with, resemble, coexist with, or be clinically related to the original.
   [Medicine] a different drug a patient could plausibly have been prescribed for a related problem.
   [Symptom] a related but clinically distinct symptom.
   [Dosage] a different amount, strength or frequency, differing by enough to change clinical meaning.
   [Test Result] a different value or finding, or a reversed result.
   [Age] an age in a clinically different band.
   [Temporal] a clinically different onset, duration or sequence.
5. Do NOT delete the fact. The document must still talk about the same detail and say something different about it. The new value must not already appear elsewhere in the document.""",
        "categories": ALTERABLE_CATEGORIES,
        "ex_fact": "[Temporal] ১০.০৯.১৯ তারিখ থেকে জ্বর শুরু হয়েছে এবং আজ ১০ দিন হল।",
        "ex_original": "১০.০৯.১৯ তারিখ থেকে ||| আজকে ১০ দিন হল",
        "ex_new": "৩১.০৮.১৯ তারিখ থেকে ||| আজকে ২০ দিন হল",
        "ex_rewritten": _DENGUE.replace("১০.০৯.১৯", "৩১.০৮.১৯").replace("আজকে ১০ দিন হল", "আজকে ২০ দিন হল"),
        "note": "The date and the day count are two mentions of one detail, so both were changed to agree. Rule 6 applies to exactly this situation.",
    },
    "COA": {
        "name": "Coarsen",
        "task": "Your task is to rewrite the source document so that it states the detail named in the target fact only VAGUELY, without the specific value it currently gives.",
        "rules": """1. Find the specific value in the document - a number, a measurement, a date, a duration, a dose - and replace it with a general expression that does not pin it down.
2. Use the natural Bangla a patient would use: মোটামুটি ভালো, একটু কম, অনেক দিন, আগের চেয়ে ভালো, স্বাভাবিকের কাছাকাছি.
3. The vague version must remain TRUE of the original value, not contradict it. If the report said ১৫০০০০, then "মোটামুটি ভালো" is correct coarsening and "খুব কম" is not - that would be a substitution, which is a different method.
4. After the rewrite the document must no longer support the specific value the summary gives, while still supporting the general direction.
5. Do NOT delete the sentence and do NOT make any other value in the document vaguer.""",
        "categories": COARSENABLE_CATEGORIES,
        "ex_fact": "[Test Result] আজকের রিপোর্টে প্লাটিলেট ১৫০০০০ আসছে।",
        "ex_original": "সর্বশেষ আজকের রিপোর্টে ১৫০০০০ আসছে।",
        "ex_new": "সর্বশেষ রিপোর্টে প্লাটিলেট মোটামুটি ভালো আসছে।",
        "ex_rewritten": _dengue_with("সর্বশেষ আজকের রিপোর্টে ১৫০০০০ আসছে।", "সর্বশেষ রিপোর্টে প্লাটিলেট মোটামুটি ভালো আসছে।"),
        "note": "The summary's specific number is now unsupported while its direction of travel is still supported. This produces partial support rather than a clean contradiction, which is the point of the method.",
    },
}


def build_prompt(source, fact, method):
    """Render the generation prompt for one (source, fact, method)."""
    spec = METHODS[method]
    body = _SKELETON.format(
        task=spec["task"],
        method_rules=spec["rules"],
        shared_rules=_SHARED_RULES,
        ex_source=_DENGUE,
        ex_fact=spec["ex_fact"],
        ex_original=spec["ex_original"],
        ex_new=spec["ex_new"],
        ex_rewritten=spec["ex_rewritten"],
    )
    if spec.get("note"):
        body = body.replace(
            "\nNow perform the edit.", "\nNote: {}\n\nNow perform the edit.".format(spec["note"])
        )
    return body.format(source=source, target_fact=format_target(fact))


# ---------------------------------------------------------------------------
# Stage 2 - job enumeration
# ---------------------------------------------------------------------------

def build_jobs(doc_id, source, facts, methods=("SUB", "COA"), seed=0):
    """Enumerate one generation job per method, for one document.

    A job is one API call. Each method draws its own target from the facts its
    category filter admits, so a document contributes at most one item per
    method and a method is skipped when the summary has no fact it can act on.
    """
    rng = random.Random("{}|{}".format(seed, doc_id))
    pool = targetable(facts, doc_id, ALTERATION_EXCLUDED, seed)

    jobs = []
    for method in methods:
        spec = METHODS[method]
        eligible = [f for f in pool if f[1] in spec["categories"]]
        if not eligible:
            continue
        target = rng.choice(eligible)
        jobs.append(
            {
                "doc_id": doc_id,
                "pipeline": "alteration",
                "operation": "alter",
                "method": method,
                "method_name": spec["name"],
                "n": 1,
                "targets": [target],
                "target_categories": [target[1]],
                "retained": [f for f in facts if f != target],
                "prompt": build_prompt(source, target, method),
            }
        )
    return jobs
