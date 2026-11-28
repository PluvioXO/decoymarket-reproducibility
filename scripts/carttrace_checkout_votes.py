"""Repeat a fixed CartTrace audit panel to test larger inference budgets."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from scripts.checkout_expansion_report import MODELS
from scripts.collect_live import write_json


PANEL_SIZE = 768
REPEATS = 3
ARMS = ("card", "reread", "receipt")


def prepare(original, output):
    data = original.read_bytes()
    panel = json.loads(data)
    cases = panel["cases"][:PANEL_SIZE]
    assert len(cases) == PANEL_SIZE
    replicas = [{**case, "id": f"{case['id']}-repeat-{replica}"}
                for replica in range(1, REPEATS + 1) for case in cases]
    write_json(output, {"original_sha256": sha256(data).hexdigest(), "cases": replicas})


def read_rows(path, limit):
    rows = {}
    for block in sorted(path.glob("block-*.json.gz")):
        with gzip.open(block, "rt") as handle:
            batch = json.load(handle)
        for row in batch:
            if row["id"] in rows:
                raise ValueError(f"Duplicate result {row['id']}")
            rows[row["id"]] = row
        if len(rows) >= limit:
            break
    return rows


def vote_counts(cases, original, repeats):
    counts = Counter()
    for case in cases:
        key = case["id"]
        rows = [original[key], *(repeats[f"{key}-repeat-{r}"] for r in range(1, REPEATS + 1))]
        assert all(row["correct"] == case["correct"] for row in rows)
        label = "right" if case["correct"] else "wrong"
        counts[label] += 1
        for arm in ARMS:
            approvals = [row["calls"][arm]["choice"] == "approve" for row in rows]
            for draws in range(1, len(approvals) + 1):
                count = sum(approvals[:draws])
                for threshold in range(1, draws + 1):
                    counts[f"{arm}_{threshold}_of_{draws}_{label}_approved"] += count >= threshold
            for row in rows:
                call = row["calls"][arm]
                counts[f"{arm}_tokens"] += call["prompt_tokens"] + call["completion_tokens"]
                counts[f"{arm}_latency_s"] += call["latency_s"]
    return counts


def frontier_comparison(models, baseline):
    counts = Counter()
    for result in models.values():
        allowed = result["receipt_1_of_1_wrong_approved"]
        policies = [f"{baseline}_{threshold}_of_{draws}" for draws in range(1, REPEATS + 2)
                    for threshold in range(1, draws + 1)]
        best = max((result[policy + "_right_approved"] for policy in policies
                    if result[policy + "_wrong_approved"] <= allowed), default=0)
        sourced = result["receipt_1_of_1_right_approved"]
        counts["receipt_more" if sourced > best else "baseline_more" if sourced < best else "tie"] += 1
    return counts


def cross_fitted_threshold(folds, arm):
    policies = {f"{arm}_{threshold}_of_{draws}": draws for draws in range(1, REPEATS + 2)
                for threshold in range(1, draws + 1)}
    policies["always_refer"] = 0
    receipt = "receipt_1_of_1"
    selected = []
    totals = Counter()
    for fit, test in (folds, folds[::-1]):
        def scores(policy):
            return fit[policy + "_right_approved"], -fit[policy + "_wrong_approved"], -policies[policy]

        allowed = fit[receipt + "_wrong_approved"]
        policy = max((name for name in policies if fit[name + "_wrong_approved"] <= allowed), key=scores)
        selected.append(policy)
        for label in ("right", "wrong"):
            totals[label] += test[label]
            totals[f"threshold_{label}_approved"] += test[f"{policy}_{label}_approved"]
            totals[f"receipt_{label}_approved"] += test[f"{receipt}_{label}_approved"]
    return {"policies_by_training_fold": selected, **totals}


def summarize(root, repeat_root, output):
    data = (root / "panel.json").read_bytes()
    repeat_panel = json.loads((repeat_root / "panel.json").read_bytes())
    assert repeat_panel["original_sha256"] == sha256(data).hexdigest()
    cases = json.loads(data)["cases"][:PANEL_SIZE]
    stores = sorted({case["store"] for case in cases}, key=lambda key: sha256(key.encode()).digest())
    fit_stores = set(stores[:len(stores) // 2])
    partitions = ([case for case in cases if case["store"] in fit_stores],
                  [case for case in cases if case["store"] not in fit_stores])
    assert all({case["correct"] for case in batch} == {True, False} for batch in partitions)
    models, cross_fitted = {}, {}
    for model in MODELS:
        baseline = root / "checkout-audit" / model / "0"
        repeat = repeat_root / "checkout-audit" / model / "0"
        first, more = read_rows(baseline, PANEL_SIZE), read_rows(repeat, len(repeat_panel["cases"]))
        if len(first) < PANEL_SIZE or len(more) != len(repeat_panel["cases"]):
            continue
        folds = tuple(vote_counts(batch, first, more) for batch in partitions)
        combined = folds[0].copy()
        combined.update(folds[1])
        models[model] = combined
        cross_fitted[model] = {arm: cross_fitted_threshold(folds, arm) for arm in ("card", "reread")}
        assert all(result["right"] + result["wrong"] == PANEL_SIZE for result in cross_fitted[model].values())
    if not models:
        raise ValueError("No complete repeated audits")
    write_json(output, {"cases_per_model": PANEL_SIZE, "seeds_per_case": REPEATS + 1, "models": models,
                        "same_or_fewer_incorrect_approvals": {baseline: frontier_comparison(models, baseline)
                                                              for baseline in ("card", "reread")},
                        "store_cross_fit": {"stores_per_fold": [len(fit_stores), len(stores) - len(fit_stores)],
                                            "cases_per_fold": [len(batch) for batch in partitions], "models": cross_fitted}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.first, args.second)
    elif args.output is not None:
        summarize(args.first, args.second, args.output)
    else:
        parser.error("summarize requires an output file")


if __name__ == "__main__":
    main()
