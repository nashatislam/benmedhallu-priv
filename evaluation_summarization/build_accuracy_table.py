# -*- coding: utf-8 -*-
"""Build the NER-style accuracy table for the summarization track. NO API calls.

Matches the layout of BenMedHalluEval - NER Evaluation:

    rows    = detector models
    columns = category x {Correct, Wrong}

    Correct = accuracy on (original source, original summary)   gold 0
    Wrong   = accuracy on (corrupted source, original summary)  gold 1

In the NER table a category's Correct column is the same sentences as its Wrong
column, unmodified. The same pairing is used here: a category's Correct set is
the untouched pair of every document that produced a corrupted item in that
category. One document can therefore appear under several categories, exactly as
one sentence can carry several entity types.

A second table repeats the split by corruption method instead of category.
"""

import glob
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

GENERATOR = sys.argv[1] if len(sys.argv) > 1 else "luna"

frames = []
for path in sorted(glob.glob(f"{GENERATOR}_summarization_evaluated_by_*.csv")):
    d = pd.read_csv(path)
    d["detector"] = path.split("evaluated_by_")[1][:-4]
    frames.append(d)
if not frames:
    raise SystemExit(f"no evaluation files found for '{GENERATOR}'")
v = pd.concat(frames, ignore_index=True)

unparseable = int(v["prediction"].isna().sum())
v = v.dropna(subset=["prediction"]).copy()
v["correct"] = v["correct"].astype(int)
v["cat"] = v["target_categories"].astype(str).str.split(" | ", regex=False).str[0]
v["variant"] = v.apply(
    lambda r: r["method"] if r["method"] not in ("Delete", "", None) and pd.notna(r["method"])
    else (f"Delete N={int(r['n'])}" if r["track"] == "B" else ""), axis=1)

track_a = v[v["track"] == "A"]
track_b = v[v["track"] == "B"]

ORDER = ["detector"]
DISPLAY = {
    "gpt_4o_mini": "GPT-4o-mini", "gemini_3_1_flash_lite": "Gemini-3.1-flash-lite",
    "grok_4_3": "Grok-4.3", "qwen_2_5_72b": "Qwen 2.5 72B Instruct",
    "llama_3_1_70b": "Llama 3.1 70B Instruct", "deepseek_v4_flash": "Deepseek V4 Flash",
}
ROWS = ["gpt_4o_mini", "gemini_3_1_flash_lite", "grok_4_3",
        "qwen_2_5_72b", "llama_3_1_70b", "deepseek_v4_flash"]


def build(split_col, out_name, title):
    """One table: rows = detectors, columns = <group> x {Correct, Wrong}."""
    groups = [g for g in track_b[split_col].dropna().unique() if str(g).strip()]
    groups = sorted(groups, key=lambda g: -int((track_b[split_col] == g).sum()))

    table = {}
    for detector in ROWS:
        b_det = track_b[track_b["detector"] == detector]
        a_det = track_a[track_a["detector"] == detector]
        row = {}
        for g in groups:
            b_g = b_det[b_det[split_col] == g]
            # Correct = the untouched pairs of the documents behind this group.
            docs = set(b_g["doc_id"])
            a_g = a_det[a_det["doc_id"].isin(docs)]
            row[(g, "Correct")] = round(100 * a_g["correct"].mean(), 2) if len(a_g) else None
            row[(g, "Wrong")] = round(100 * b_g["correct"].mean(), 2) if len(b_g) else None
        row[("Overall", "Correct")] = round(100 * a_det["correct"].mean(), 2)
        row[("Overall", "Wrong")] = round(100 * b_det["correct"].mean(), 2)
        table[DISPLAY[detector]] = row

    df = pd.DataFrame(table).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "Model"

    flat = df.copy()
    flat.columns = ["{}_{}".format(a, b) for a, b in df.columns]
    flat.to_csv(out_name, encoding="utf-8-sig")

    counts = {g: int((track_b[split_col] == g).sum() / len(ROWS)) for g in groups}
    print("=" * 100)
    print(title)
    print("items per group (Wrong):", ", ".join(f"{g}={counts[g]}" for g in groups),
          f"| Correct pairs={int(len(track_a) / len(ROWS))}")
    print("=" * 100)
    print(df.to_string())
    print()
    print("saved:", out_name)
    print()
    return df


print(f"generator: {GENERATOR}   unparseable verdicts excluded: {unparseable}")
print()
build("cat", f"{GENERATOR}_accuracy_by_category.csv",
      "ACCURACY BY FACT CATEGORY  (Correct = original source, Wrong = corrupted source)")
build("variant", f"{GENERATOR}_accuracy_by_method.csv",
      "ACCURACY BY CORRUPTION METHOD  (Correct = original source, Wrong = corrupted source)")
