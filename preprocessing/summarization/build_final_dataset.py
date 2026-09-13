# -*- coding: utf-8 -*-
"""Merge the two pipeline files into one evaluation-ready dataset. NO API calls.

Drops rows whose gold label cannot be right:

  * source came back byte-identical to the original - nothing was corrupted, so
    the summary is still faithful and the "hallucinated" label is wrong;
  * a deletion that did not shrink the document - the removal silently failed.

Both checks are free and need no model. They are the same two the detector
evaluation used as a pre-filter, applied once here instead.

Output columns:
    item_id            stable id for joining verdicts back
    doc_id, split      provenance
    pipeline, method, n
    target_categories, target_facts, retained_facts
    original_span, new_span          gold spans for the localisation task
    source_original                  Track A input, gold 0
    source_corrupted                 Track B input, gold 1
    summary                          shared by both tracks
"""

import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = pd.read_csv("full_luna_deletion.csv")
a = pd.read_csv("full_luna_alteration.csv")
d["pipeline"], a["pipeline"] = "deletion", "alteration"
a["n"] = 1                      # alteration is always single-fact
d["new_span"] = pd.NA           # deletion never writes a replacement

merged = pd.concat([d, a], ignore_index=True)
before = len(merged)

orig = merged["source_original"].astype(str).str.strip()
corr = merged["source_corrupted"].astype(str).str.strip()
unchanged = corr == orig
not_shrunk = (merged["pipeline"] == "deletion") & (corr.str.len() >= orig.str.len())
bad = unchanged | not_shrunk

print("DROPPED ROWS")
print("  source unchanged                 ", int(unchanged.sum()))
print("  deletion did not shrink          ", int(not_shrunk.sum()))
print("  overlap between the two          ", int((unchanged & not_shrunk).sum()))
print("  total dropped                    ", int(bad.sum()))
print()
print("  by variant:")
v = merged.loc[bad].apply(
    lambda r: r["method"] if r["pipeline"] == "alteration" else f"Delete N={int(r['n'])}", axis=1)
for k, n in v.value_counts().items():
    print(f"     {k:<14} {n}")

merged = merged[~bad].reset_index(drop=True)
merged["item_id"] = [
    "{}_{}_{}".format(r["doc_id"], r["method"], int(r["n"]))
    for _, r in merged.iterrows()
]
assert merged["item_id"].is_unique, "item_id collision"

COLS = ["item_id", "doc_id", "split", "pipeline", "method", "n",
        "target_categories", "target_facts", "retained_facts",
        "original_span", "new_span",
        "source_original", "source_corrupted", "summary"]
merged = merged[COLS]

OUT = "final_luna_corrupted.csv"
merged.to_csv(OUT, index=False, encoding="utf-8-sig")

# Track A stays a separate deduplicated file: the faithful pair repeats across
# every variant of a document, so scoring it inline would pay for the same call
# many times over.
track_a = (merged.drop_duplicates("doc_id")[["doc_id", "split", "source_original", "summary"]]
           .reset_index(drop=True))
track_a.to_csv("final_luna_track_a.csv", index=False, encoding="utf-8-sig")

print()
print("=" * 64)
print(f"{OUT:<32} {len(merged):>6} rows  ({before} - {int(bad.sum())})")
print(f"{'final_luna_track_a.csv':<32} {len(track_a):>6} rows")
print("=" * 64)
print()
print("by split:")
print(merged.groupby(["split", "pipeline"]).size().unstack(fill_value=0).to_string())
print()
print("by variant:")
vv = merged.apply(
    lambda r: r["method"] if r["pipeline"] == "alteration" else f"Delete N={int(r['n'])}", axis=1)
print(vv.value_counts().sort_index().to_string())
print()
print("nulls in the columns the prompt uses:",
      {c: int(merged[c].isna().sum()) for c in
       ("source_original", "source_corrupted", "summary")})
print("documents covered:", merged["doc_id"].nunique())
print()
print(f"evaluation calls if all six detectors score everything: "
      f"{6 * (len(merged) + len(track_a)):,}")
print(f"test split only: {6 * (len(merged[merged['split'] == 'test']) + len(track_a[track_a['split'] == 'test'])):,}")
