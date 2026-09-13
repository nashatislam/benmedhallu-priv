# -*- coding: utf-8 -*-
"""
Shared stages for both source corruption pipelines.

The inversion: the human summary is NEVER edited. We corrupt the SOURCE
document, so the untouched gold summary stops being supported by it. Every
summary in the benchmark is human-written, faithful and hallucinated items share
one text distribution, and no detector can win by spotting generated prose.

Two generation pipelines, sharing stage 1:

    Stage 1  EXTRACT_FACTS_PROMPT      summary            -> atomic facts (typed)
    Stage 2  <pipeline>.build_jobs     facts              -> one job per item
    Stage 3  b1_deletion               source, fact       -> claim becomes UNSUPPORTED
             b2_alteration             source, fact, meth -> claim becomes CONTRADICTED

Stage 4 verification lives in verification.py and is PARKED for the first run.
Generate first, read the output by hand, wire the filter in afterwards.

Selection is done in CODE, not by the model. The prompt is told exactly which
fact to act on, the same way the NER notebooks pass a specific entity into the
generation prompt. Letting the model choose its own target would cost you the
gold record of what changed, the balance across entity categories, and the
difficulty dial.
"""

import random
import re


# Categories excluded from targeting. Excluded facts are still EXTRACTED and
# still sit in every job's `retained` list, so collateral damage to them stays
# detectable at verification.
#
#   Specialist        one instance in 50 pilot documents.
#   Medical Procedure half mislabelled - "the patient gargled", "the patient
#                     saw a doctor" - and 5 facts in 50 documents.
#
# The two pipelines differ on one category, deliberately. Deletion only needs a
# fact to be locatable and removable, so entanglement does not break the label:
# removing either member of an entangled pair leaves the summary's claim
# unsupported, and both outcomes are correct. Alteration needs a well-defined
# value slot to swap or vague out, and Other facts have spans but no slots -
# "the paracetamol did not help" has nothing to replace like for like. In the
# pilot all five deletion items targeting Other succeeded.
DELETION_EXCLUDED = ("Specialist", "Medical Procedure")
ALTERATION_EXCLUDED = ("Specialist", "Medical Procedure", "Other")

# Symptom is 47% of all extracted facts and appears in 47 of 50 pilot summaries.
# Left uncapped it swamps every marginal you report. One target per document
# brings it to roughly a quarter of all targets.
CAPPED_CATEGORIES = {"Symptom": 1}


def targetable(facts, doc_id, excluded, seed=0):
    """The facts a pipeline may corrupt, after its exclusions and the shared cap.

    Deterministic per document. The cap is resolved against the FULL fact list
    before exclusions are applied, so both pipelines keep the same symptom fact
    even though their exclusion sets differ.
    """
    rng = random.Random("cap|{}|{}".format(seed, doc_id))

    keep = list(range(len(facts)))
    for category, limit in CAPPED_CATEGORIES.items():
        positions = [i for i, f in enumerate(facts) if f[1] == category]
        if len(positions) > limit:
            chosen = set(rng.sample(positions, limit))
            keep = [i for i in keep if facts[i][1] != category or i in chosen]

    return [facts[i] for i in keep if facts[i][1] not in excluded]


# Superset of the seven NER columns in dataset/NER/test_tag_columns.csv, plus
# three categories that clinical narrative needs but entity tagging does not
# cover. Keeping the first seven spelled exactly as the CSV columns lets the
# summarization results be reported against the NER track per category.
FACT_CATEGORIES = [
    "Symptom",
    "Health Condition",
    "Medicine",
    "Specialist",
    "Age",
    "Dosage",
    "Medical Procedure",
    "Test Result",
    "Temporal",
    "Other",
]


# ---------------------------------------------------------------------------
# Stage 1 - decompose the gold summary into atomic, orthogonal facts
# ---------------------------------------------------------------------------

EXTRACT_FACTS_PROMPT = """You are given a Bangla medical summary written by a human annotator.

Your task is to decompose the summary into atomic facts.

Rules:
1. An atomic fact states exactly ONE piece of information and cannot be split further without losing meaning.
2. Decompose compound statements. "রোগীর বয়স ৩৫ এবং তার উচ্চ রক্তচাপ আছে" contains two atomic facts, not one.
3. The facts must be ORTHOGONAL: no two facts may refer to the same underlying clinical detail, and changing one must not change another. If two details always move together, such as a start date and the number of days since, record them as ONE fact.
4. Write each fact in Bangla as a short standalone sentence that is understandable without reading the summary.
5. Extract facts ONLY from the summary. Do not add information, do not infer, do not consult outside medical knowledge.
6. Do not extract questions the patient asks. Extract only asserted information.
7. Do not extract hedges, greetings, or requests for advice.
8. Assign each fact exactly ONE category from this list:
   Symptom, Health Condition, Medicine, Specialist, Age, Dosage, Medical Procedure, Test Result, Temporal, Other
9. Use Temporal for onset, duration, and sequence. Use Test Result for reported findings and values. Use Dosage for amounts, strengths, and frequencies of treatment.
10. Number the facts sequentially starting from 1.
11. Do not provide explanations, commentary, or a total count.
12. Return your answer in EXACTLY the following format, one fact per line:

FACT <number> | <category> | <atomic fact in Bangla>

Example:

Summary:
আমার ১০.০৯.১৯ তারিখ থেকে ডেঙ্গু জ্বর, আজকে ১০ দিন হল। এনএস১ পজিটিভ আসছে। প্লাটিলেট কমে যাচ্ছিল, ৩ দিন যাবত প্লাটিলেট বাড়ছে। আজকের রিপোর্টে ১৫০০০০ আসছে। শরীর কিছুটা দুর্বল।

Output:

FACT 1 | Health Condition | রোগীর ডেঙ্গু জ্বর হয়েছে।
FACT 2 | Temporal | ১০.০৯.১৯ তারিখ থেকে জ্বর শুরু হয়েছে এবং আজ ১০ দিন হল।
FACT 3 | Test Result | এনএস১ পরীক্ষার ফল পজিটিভ আসছে।
FACT 4 | Test Result | আগে প্লাটিলেট কমে যাচ্ছিল।
FACT 5 | Test Result | গত ৩ দিন যাবত প্লাটিলেট বাড়ছে।
FACT 6 | Test Result | আজকের রিপোর্টে প্লাটিলেট ১৫০০০০ আসছে।
FACT 7 | Symptom | রোগীর শরীর কিছুটা দুর্বল।

Now decompose the following summary.

Summary:
{summary}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_facts(response):
    """Parse Stage 1 output into [(index, category, fact_text), ...]."""
    facts = []
    for line in response.strip().splitlines():
        match = re.match(r"^\s*FACT\s+(\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$", line)
        if match:
            index, category, text = match.groups()
            facts.append((int(index), category.strip(), text.strip()))
    return facts


def parse_rewritten_source(response):
    """Parse the rewritten document. Returns None when the model reported failure."""
    match = re.search(r"Rewritten Source:\s*(.+)", response, re.DOTALL)
    if not match:
        return None
    body = match.group(1).strip()
    if not body or "FAILED" in body:
        return None
    return body


def parse_spans(response):
    """Parse the gold span lines the rewrite prompts emit.

    Returns (original_span, new_span). These are what the span localisation task
    scores against, and they cost nothing because the edit was controlled, so
    write them to the CSV alongside the rewritten document. original_span is
    None when the method added text rather than replacing any.
    """
    original = re.search(r"Original Span:\s*(.+)", response)
    new = re.search(r"New Span:\s*(.+)", response)
    original_text = original.group(1).strip() if original else None
    if original_text in (None, "", "NONE"):
        original_text = None
    return original_text, new.group(1).strip() if new else None


def format_target(fact):
    """Render one selected fact as the typed line the rewrite prompts expect."""
    _, category, text = fact
    return "[{}] {}".format(category, text)


def format_targets(facts):
    """Render several selected facts as a numbered, typed block."""
    return "\n".join(
        "{}. [{}] {}".format(i, category, text)
        for i, (_, category, text) in enumerate(facts, start=1)
    )
