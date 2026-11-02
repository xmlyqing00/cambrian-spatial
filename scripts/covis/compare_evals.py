#!/usr/bin/env python
"""Paired comparison of two lmms-eval runs on the same questions (sample logs).

Usage: python scripts/covis/compare_evals.py BASE_RUN NEW_RUN
A run is a sample log (e.g. vsibench.json, cambrians_vsr_10mins.json) or a directory containing one
(searched recursively; results.json is ignored). Works for any task whose per-sample score dict has
"accuracy" or "MRA:.5:.95:.05"; non-VSI-Bench tasks are grouped by doc["type"] when present.
Per question: accuracy (multiple choice) or MRA (numerical). Reports both runs' VSI-Bench overall
(official aggregation: mean over question types, rel_direction = mean of its three subtypes), the
paired per-question mean difference with a bootstrap 95% CI, an exact McNemar test on
multiple-choice questions, per-type means, and how many predictions are identical.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def load(run):
    if run.endswith(".json"):
        fp = run
    else:
        fps = [f for f in glob.glob(os.path.join(run, "**", "*.json"), recursive=True) if os.path.basename(f) != "results.json"]
        fp = sorted(fps, key=lambda f: (os.path.basename(f) != "vsibench.json", f))[0]
    out = {}
    for l in json.load(open(fp))["logs"]:
        dicts = [v for k, v in l.items() if k != "doc" and isinstance(v, dict)] + [l["doc"]]
        s = next(d for d in dicts if "accuracy" in d or "MRA:.5:.95:.05" in d)
        score = s["accuracy"] if "accuracy" in s else s["MRA:.5:.95:.05"]
        qtype = s.get("question_type") or l["doc"].get("type") or "all"
        out[l["doc_id"]] = dict(q=l["doc"]["question"], type=qtype, mca="accuracy" in s, score=float(score), pred=str(s["prediction"]))
    return fp, out


def overall(rows):
    """Official VSI-Bench aggregation: mean per question type, rel_direction = mean of its 3 subtypes."""
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r["score"])
    means = {t: np.mean(v) for t, v in by.items()}
    sub = [t for t in means if t.startswith("object_rel_direction_")]
    if sub:
        means["object_rel_direction"] = np.mean([means.pop(t) for t in sub])
    return 100 * np.mean(list(means.values())), means, {t: len(v) for t, v in by.items()}


def main():
    (fa, a), (fb, b) = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a) & set(b))
    assert all(a[i]["q"] == b[i]["q"] for i in ids), "runs do not contain the same questions"
    print(f"base: {fa}\nnew:  {fb}\n{len(ids)} paired questions ({len(a)} / {len(b)} in runs)")
    oa, mean_a, n_by = overall([a[i] for i in ids])
    ob, mean_b, _ = overall([b[i] for i in ids])
    d = np.array([b[i]["score"] - a[i]["score"] for i in ids])
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    same = sum(a[i]["pred"] == b[i]["pred"] for i in ids)
    print(f"overall (mean over types): base {oa:.2f}  new {ob:.2f}  diff {ob - oa:+.2f}")
    print(f"per-question mean diff {100 * d.mean():+.2f} pts, bootstrap 95% CI [{100 * np.percentile(boot, 2.5):+.2f}, {100 * np.percentile(boot, 97.5):+.2f}]")
    print(f"identical predictions: {same}/{len(ids)}")
    mca = [i for i in ids if a[i]["mca"]]
    wins = sum(b[i]["score"] > a[i]["score"] for i in mca)
    losses = sum(b[i]["score"] < a[i]["score"] for i in mca)
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    print(f"multiple choice ({len(mca)} q): new right/base wrong {wins}, base right/new wrong {losses}, exact McNemar p={p:.3f}")
    print(f"{'type':28s} {'n':>4s} {'base':>7s} {'new':>7s} {'diff':>7s}")
    for t in sorted(mean_a):
        n = sum(v for k, v in n_by.items() if k == t or k.startswith(t + "_"))
        print(f"{t:28s} {n:4d} {100 * mean_a[t]:7.2f} {100 * mean_b[t]:7.2f} {100 * (mean_b[t] - mean_a[t]):+7.2f}")


if __name__ == "__main__":
    main()
