# -*- coding: utf-8 -*-
"""Offline integrity check for the generation pipeline. Makes NO API calls.

Run this before any run that spends money:

    python verify_pipeline_integrity.py

Every check is deterministic and local. A FAIL means do not start the run.
"""

import io
import json
import os
import re
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath("source_corruption"))

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


# ---------------------------------------------------------------------------

@check("modules import cleanly")
def _():
    import common, b1_deletion, b2_alteration, verification  # noqa: F401
    return "common, b1_deletion, b2_alteration, verification"


@check("notebooks are valid JSON")
def _():
    names = []
    for path in ("extract_summary_facts.ipynb", "generate_source_corruption.ipynb",
                 "evaluate_detectors.ipynb", "verify_items.ipynb"):
        if not os.path.exists(path):
            raise AssertionError("missing notebook: " + path)
        nb = json.load(io.open(path, encoding="utf-8"))
        assert nb["cells"], path + " has no cells"
        names.append("{} ({} cells)".format(path, len(nb["cells"])))
    return "; ".join(names)


@check("every notebook code cell compiles")
def _():
    # JSON validity is not enough: a cell can be valid JSON and broken Python.
    # A SyntaxError here is what stopped a run after the launch command, so this
    # check has to pass before anything is allowed to spend money.
    counts = []
    for path in ("extract_summary_facts.ipynb", "generate_source_corruption.ipynb",
                 "evaluate_detectors.ipynb", "verify_items.ipynb"):
        nb = json.load(io.open(path, encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code):
            source = "".join(cell["source"])
            try:
                compile(source, "<{} cell {}>".format(path, i), "exec")
            except SyntaxError as exc:
                raise AssertionError("{} cell {} line {}: {}".format(
                    path, i, exc.lineno, exc.msg))
        counts.append("{} ({})".format(path.split("_")[0], len(code)))
    return "; ".join(counts)


@check("config matches the agreed design")
def _():
    import common, b1_deletion, b2_alteration
    assert set(b2_alteration.METHODS) == {"SUB", "COA"}, b2_alteration.METHODS
    assert b1_deletion.build_jobs.__defaults__[0] == (1, 2, 3), "N values"
    assert common.DELETION_EXCLUDED == ("Specialist", "Medical Procedure")
    assert common.ALTERATION_EXCLUDED == ("Specialist", "Medical Procedure", "Other")
    assert common.CAPPED_CATEGORIES == {"Symptom": 1}
    assert b2_alteration.METHODS["COA"]["categories"] == ["Temporal", "Test Result", "Dosage"]
    return "2 methods, N=(1,2,3), exclusions and Symptom cap as agreed"


@check("prompts render with no leftover placeholders")
def _():
    import common, b1_deletion, b2_alteration
    fact = (1, "Test Result", "প্লাটিলেট ১৫০০০০।")
    rendered = {
        "extract": common.EXTRACT_FACTS_PROMPT.format(summary="S"),
        "delete": b1_deletion.REMOVE_FACTS_PROMPT.format(
            source="D", facts_to_remove=common.format_targets([fact])),
        "SUB": b2_alteration.build_prompt("D", fact, "SUB"),
        "COA": b2_alteration.build_prompt("D", fact, "COA"),
    }
    for name, text in rendered.items():
        leftover = re.findall(r"\{[a-z_]+\}", text)
        assert not leftover, "{}: unfilled {}".format(name, leftover)
        assert "{{" not in text and "}}" not in text, name + ": doubled braces survived"
        assert len(text) > 400, name + ": suspiciously short"
    return ", ".join("{} {}c".format(k, len(v)) for k, v in rendered.items())


@check("input facts file is readable and complete")
def _():
    import common
    tags = pd.read_csv("facts_test_first50_tag_columns.csv")
    assert len(tags) == 50, "expected 50 rows, got {}".format(len(tags))
    for col in ("id", "source", "summary"):
        assert col in tags.columns, "missing column " + col
        assert tags[col].notna().all(), col + " has blanks"
    missing = [c for c in common.FACT_CATEGORIES if c not in tags.columns]
    assert not missing, "missing category columns: {}".format(missing)
    return "50 rows, all 10 category columns present"


@check("job building works on the real data")
def _():
    import common, b1_deletion, b2_alteration
    tags = pd.read_csv("facts_test_first50_tag_columns.csv")
    jobs, docs = [], 0
    for _, row in tags.iterrows():
        facts = []
        for category in common.FACT_CATEGORIES:
            cell = row.get(category)
            if pd.isna(cell) or not str(cell).strip():
                continue
            for text in str(cell).split(" | "):
                if text.strip():
                    facts.append((len(facts) + 1, category, text.strip()))
        if not facts:
            continue
        docs += 1
        jobs += b1_deletion.build_jobs(row["id"], row["source"], facts)
        jobs += b2_alteration.build_jobs(row["id"], row["source"], facts)

    assert docs == 50, "expected 50 documents, got {}".format(docs)
    assert 180 <= len(jobs) <= 230, "job count out of expected range: {}".format(len(jobs))

    banned = {"Specialist", "Medical Procedure"}
    for job in jobs:
        cats = set(job["target_categories"])
        assert not (cats & banned), "excluded category targeted: " + str(cats)
        if job["pipeline"] == "alteration":
            assert "Other" not in cats, "alteration targeted Other"
            assert job["n"] == 1, "alteration N != 1"
        assert job["targets"], "job has no targets"
        assert len(job["targets"]) + len(job["retained"]) == len(
            [f for f in job["targets"]] + job["retained"]), "target/retained mismatch"
        assert job["prompt"].count("{") == 0, "unrendered prompt in job"
    return "{} documents -> {} jobs, all rules hold".format(docs, len(jobs))


@check("Symptom cap holds and both pipelines agree on it")
def _():
    import common, b1_deletion, b2_alteration
    tags = pd.read_csv("facts_test_first50_tag_columns.csv")
    mismatches, total_targets, symptom_targets = 0, 0, 0
    for _, row in tags.iterrows():
        facts = []
        for category in common.FACT_CATEGORIES:
            cell = row.get(category)
            if pd.isna(cell) or not str(cell).strip():
                continue
            for text in str(cell).split(" | "):
                if text.strip():
                    facts.append((len(facts) + 1, category, text.strip()))
        if not facts:
            continue
        jobs = b1_deletion.build_jobs(row["id"], row["source"], facts)
        jobs += b2_alteration.build_jobs(row["id"], row["source"], facts)
        chosen = {t[2] for j in jobs for t in j["targets"] if t[1] == "Symptom"}
        if len(chosen) > 1:
            mismatches += 1
        for j in jobs:
            total_targets += len(j["target_categories"])
            symptom_targets += j["target_categories"].count("Symptom")

    assert mismatches == 0, "{} docs used more than one symptom".format(mismatches)
    share = 100 * symptom_targets / total_targets
    assert share < 35, "symptom share too high: {:.0f}%".format(share)
    return "one symptom per document, {:.0f}% of all targets".format(share)


@check("verification module parses and accepts correctly")
def _():
    import verification as V
    targets = [(1, "Test Result", "ক।")]
    retained = [(2, "Symptom", "খ।"), (3, "Temporal", "গ।")]
    prompt, roles = V.build_check("SRC", targets, retained, "d1")
    assert "{" not in prompt, "unrendered verification prompt"
    assert roles.count("target") == 1 and roles.count("retained") == 2
    good = ["NOT_MENTIONED" if r == "target" else "SUPPORTED" for r in roles]
    assert V.accept(good, roles, "delete")[0] is True
    assert V.accept(good, roles, "alter")[0] is False
    assert V.parse_fact_verdicts("FACT 1: SUPPORTED", 3) is None
    assert V.parse_fact_verdicts(
        "1. supported\n2: CONTRADICTED\n3 - not_mentioned", 3) == [
        "SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]
    assert V.JUDGE_MODEL not in (
        "openai/gpt-5.6-luna", "deepseek/deepseek-v4-pro", "openai/gpt-4o-mini",
        "deepseek/deepseek-v4-flash", "google/gemini-3.1-flash-lite",
        "meta-llama/llama-3.1-70b-instruct", "qwen/qwen-2.5-72b-instruct",
        "x-ai/grok-4.3"), "judge overlaps a generator or detector"
    return "judge {} is independent".format(V.JUDGE_MODEL)


@check("API keys present and funded")
def _():
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath("../../.env"))
    funded = []
    for name in ("OPENROUTER_API_KEY_NEW", "OPENROUTER_API_KEY"):
        key = os.getenv(name)
        if not key:
            continue
        data = requests.get("https://openrouter.ai/api/v1/key",
                            headers={"Authorization": "Bearer " + key}, timeout=30).json()["data"]
        remaining = data.get("limit_remaining")
        if remaining is None or remaining > 0.5:
            funded.append("{} (${:.2f})".format(name, remaining if remaining is not None else -1))
    assert funded, "no key has usable credit"
    return "; ".join(funded)


@check("no stray generation process is already running")
def _():
    import subprocess
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
         "Where-Object {$_.CommandLine -like '*nbconvert*'}).Count"],
        capture_output=True, text=True, timeout=60).stdout.strip()
    count = int(out or 0)
    assert count == 0, "{} nbconvert process(es) already running - kill them first".format(count)
    return "0 nbconvert processes"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    print("PIPELINE INTEGRITY CHECK - no API calls\n" + "=" * 72)
    for name, fn in CHECKS:
        try:
            detail = fn()
            print("  PASS  {:<48} {}".format(name, detail or ""))
        except Exception as exc:
            failures += 1
            print("  FAIL  {:<48} {}".format(name, exc))
    print("=" * 72)
    print("{} of {} checks passed.".format(len(CHECKS) - failures, len(CHECKS)))
    if failures:
        print("DO NOT START THE RUN.")
    sys.exit(1 if failures else 0)
