# -*- coding: utf-8 -*-
"""Merge the four pilot CSVs into two evaluation sets, one per generator.

Makes NO API calls.

Each output row carries the three things the detector prompt uses - the original
source, the corrupted source, and the untouched human summary - plus the labels
needed to break results down afterwards.

One row yields TWO detector calls, which is the dual track:

    Track A   (source_original,  summary)  ->  gold 0, faithful
    Track B   (source_corrupted, summary)  ->  gold 1, hallucinated

Track A pairs are identical across generators and repeat across the variants of
one document, so they are deduplicated to one row per document and written once
to a shared file. Scoring them per generator would pay for the same call twice.
"""

import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GENERATORS = ("luna", "deepseek")
CONTENT = ["source_original", "source_corrupted", "summary"]
LABELS = ["doc_id", "method", "n", "target_categories"]

track_a_rows = {}
print("%-32s %5s %6s" % ("file", "rows", "cols"))
print("-" * 46)

for gen in GENERATORS:
    frames = []
    for pipeline in ("deletion", "alteration"):
        f = pd.read_csv("pilot_{}_{}.csv".format(gen, pipeline))
        if "n" not in f.columns:
            f["n"] = 1
        f["method"] = f["method"].fillna("Delete")
        frames.append(f[LABELS + CONTENT])
    merged = pd.concat(frames, ignore_index=True).sort_values(["doc_id", "method", "n"])
    merged = merged.reset_index(drop=True)

    out = "eval_{}.csv".format(gen)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print("%-32s %5d %6d" % (out, len(merged), len(merged.columns)))

    for _, row in merged.iterrows():
        track_a_rows.setdefault(row["doc_id"], (row["source_original"], row["summary"]))

track_a = pd.DataFrame(
    [{"doc_id": k, "source_original": v[0], "summary": v[1]}
     for k, v in track_a_rows.items()]
).sort_values("doc_id").reset_index(drop=True)
track_a.to_csv("eval_track_a.csv", index=False, encoding="utf-8-sig")
print("%-32s %5d %6d" % ("eval_track_a.csv", len(track_a), len(track_a.columns)))

print()
print("columns kept:", LABELS + CONTENT)
print()
print("The detector prompt uses ONLY a source and the summary. The label columns")
print("exist so results can be broken down by method, by N and by fact category,")
print("which is the per-category reporting the NER track uses.")
print()
calls_b = sum(len(pd.read_csv("eval_{}.csv".format(g))) for g in GENERATORS)
print("detector calls per model: %d (Track B) + %d (Track A) = %d"
      % (calls_b, len(track_a), calls_b + len(track_a)))
print("across 6 detectors: %d calls" % (6 * (calls_b + len(track_a))))
