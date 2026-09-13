# -*- coding: utf-8 -*-
"""
Stage 4: verification. Run this BEFORE the detector evaluation.

Verification checks the DATA; the detector evaluation measures the MODELS. The
order matters: if a deletion silently failed and the summary is still supported,
the item is labelled hallucinated, every detector correctly says faithful, and
all six take a penalty for being right.

One batched call per item instead of one call per fact. The judge sees the
corrupted source and every fact from the summary - targeted and retained
together, shuffled - and returns a verdict per line. Acceptance then reads the
list:

    deletion    every targeted fact must be NOT_MENTIONED
    alteration  every targeted fact must be CONTRADICTED
    both        every retained fact must be SUPPORTED

A targeted fact returning the OTHER pipeline's verdict means the model applied
the wrong edit, and the item belongs in the other pool rather than being a
plain failure. accept() reports that separately.

Why the statements are shuffled: the judge must not be able to infer which
facts were targeted from their position in the list.

THE JUDGE MUST BE INDEPENDENT. Not the generator, which would rubber-stamp its
own edits, and not a member of the detector panel, which would make the
benchmark circular. BanglaSummEval names this self-reinforcing loop as its own
limitation, so a reviewer will look for it.

    generator  openai/gpt-5.6-luna, deepseek/deepseek-v4-pro
    detectors  openai/gpt-4o-mini, deepseek/deepseek-v4-flash,
               google/gemini-3.1-flash-lite, meta-llama/llama-3.1-70b-instruct,
               qwen/qwen-2.5-72b-instruct, x-ai/grok-4.3
    judge      anthropic/claude-haiku-4.5   <- outside every family above

Cost: roughly 2 calls per item (facts + naturalness), about 1,100 input and 100
output tokens each. At Haiku 4.5 rates that is around $0.0016 per call, so a
200-item set verifies for well under a dollar.
"""

import random
import re


JUDGE_MODEL = "anthropic/claude-haiku-4.5"


# ---------------------------------------------------------------------------
# The batched fact check
# ---------------------------------------------------------------------------

VERIFY_FACTS_PROMPT = """You are a medical verification assistant.

You are given a Bangla medical document and a numbered list of statements.

For each statement, determine its relationship to the document.

Rules:
1. Judge ONLY against the document. Do not use outside medical knowledge, and do not consider whether the statement is medically plausible in general.
2. A statement the document plainly licenses as an inference counts as SUPPORTED. You do not need the exact words.
3. A statement is CONTRADICTED only when the document asserts something that cannot be true at the same time as the statement. A document that is simply silent on the matter is NOT_MENTIONED, not CONTRADICTED.
4. A statement is NOT_MENTIONED when the document neither supports nor contradicts it, including when the document once discussed the topic but no longer gives the detail the statement claims.
5. Judge each statement independently. Do not let the verdict on one statement influence another, and do not assume the statements agree with each other.
6. Return exactly one line per statement, in the same order they were given, and nothing else. No explanation, no preamble, no summary.

Document:
{source}

Statements:
{statements}

Answer in EXACTLY this format, one line per statement:
FACT <number>: <SUPPORTED or CONTRADICTED or NOT_MENTIONED>
"""


VERIFY_NATURAL_PROMPT = """You are a native Bangla speaker reviewing a patient's message to an online health forum.

Determine whether the text reads naturally, as something a real patient would write.

Text:
{source}

Output 1 if the text is coherent and reads like genuine patient writing.
Output 0 if it has an abrupt gap, a dangling reference, a broken sentence, an internal contradiction, or reads as if something was edited out of it.

Answer in EXACTLY this format, with no explanation:
ANSWER: <0 or 1>
"""


# ---------------------------------------------------------------------------
# Building one verification call
# ---------------------------------------------------------------------------

def build_check(corrupted_source, targets, retained, doc_id, seed=0):
    """Render the batched prompt for one item.

    Returns (prompt, roles) where roles[i] is "target" or "retained" for the
    statement on line i+1. Shuffled, so position leaks nothing to the judge.
    """
    entries = [("target", f) for f in targets] + [("retained", f) for f in retained]
    random.Random("verify|{}|{}".format(seed, doc_id)).shuffle(entries)

    statements = "\n".join(
        "{}. {}".format(i, fact[2]) for i, (_, fact) in enumerate(entries, start=1)
    )
    prompt = VERIFY_FACTS_PROMPT.format(source=corrupted_source, statements=statements)
    return prompt, [role for role, _ in entries]


# ---------------------------------------------------------------------------
# Parsing and acceptance
# ---------------------------------------------------------------------------

VERDICTS = ("SUPPORTED", "CONTRADICTED", "NOT_MENTIONED")


def parse_fact_verdicts(response, expected):
    """Parse the batched reply into a list of `expected` verdicts.

    Returns None when the reply is unusable - wrong count, unparseable lines -
    which is itself a reject reason rather than something to paper over.
    """
    if not response:
        return None

    found = {}
    for line in response.splitlines():
        match = re.match(
            r"^\s*(?:FACT\s+)?(\d+)\s*[:.\-]\s*(%s)\b" % "|".join(VERDICTS),
            line.strip(), re.IGNORECASE,
        )
        if match:
            found[int(match.group(1))] = match.group(2).upper()

    if len(found) != expected or set(found) != set(range(1, expected + 1)):
        return None
    return [found[i] for i in range(1, expected + 1)]


def parse_binary(response):
    """Parse the naturalness verdict. Returns 1, 0, or None."""
    if not response:
        return None
    match = re.search(r"ANSWER:\s*([01])", response)
    return int(match.group(1)) if match else None


def accept(verdicts, roles, operation):
    """Acceptance test for one built item.

    operation is "delete" or "alter". Returns (ok, reason). The reason strings
    are worth keeping in the output CSV: the rejection profile tells you which
    prompt rule is not landing.
    """
    if verdicts is None:
        return False, "unparseable verdict block"

    want = "NOT_MENTIONED" if operation == "delete" else "CONTRADICTED"
    other = "CONTRADICTED" if operation == "delete" else "NOT_MENTIONED"

    targeted = [v for v, r in zip(verdicts, roles) if r == "target"]
    retained = [v for v, r in zip(verdicts, roles) if r == "retained"]

    if any(v == other for v in targeted):
        return False, "wrong operation applied, item belongs in the other pool"
    if any(v == "SUPPORTED" for v in targeted):
        return False, "targeted fact still supported, the edit did not take"
    if any(v != want for v in targeted):
        return False, "targeted fact has an unexpected verdict"
    if any(v != "SUPPORTED" for v in retained):
        return False, "collateral damage to an untargeted fact"
    return True, "ok"
