"""Report receipt coverage and paired audit outcomes across held-out stores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import quantiles

from scripts.carttrace_checkout_audit import ARMS, REVISION, update
from scripts.checkout_expansion_report import MODELS
from scripts.collect_live import write_json


def pricing(cases):
    counts = Counter()
    for case in cases:
        receipt = case["receipt"]
        prices = [line.get("listed_price") for line in case["listing"]]
        if not all(isinstance(p, dict) and p.get("currency") == receipt["currency"] and type(p.get("amount")) is int for p in prices):
            continue
        headline = sum(p["amount"] * line["quantity"] for p, line in zip(prices, case["listing"], strict=True))
        total = receipt["charges"]["total"]
        counts.update(bound_listed_total=1, higher_total=total > headline, same_total=total == headline, lower_total=total < headline)
        for pct in (0, 5, 10, 20, 40):
            counts[f"budget_exceeded_at_{pct}_percent_margin"] += total * 100 > headline * (100 + pct)
    return counts


def store_interval(stores, key):
    pairs = list(stores.values())
    samples = [(v[key], v["count"]) for v in pairs if v["count"]]
    if len(samples) < 2:
        return None
    rng = random.Random(4651)
    values = []
    for _ in range(4000):
        selected = rng.choices(samples, k=len(samples))
        values.append(sum(num for num, _ in selected) / sum(n for _, n in selected))
    cut = quantiles(values, n=40, method="inclusive")
    return {"delta_percentage_points": 100 * sum(num for num, _ in samples) / sum(n for _, n in samples),
            "paired_store_bootstrap_95_points": [100 * cut[0], 100 * cut[-1]], "stores": len(samples)}


def report(root, max_cases, split="test_cross_store"):
    data = (root / "panel.json").read_bytes()
    panel = json.loads(data)
    if panel["dataset_revision"] != REVISION or panel["split"] != split:
        raise ValueError(f"Use the pinned {split} checkout panel")
    cases = panel["cases"]
    paths = {m: root / "checkout-audit" / m / "0" for m in MODELS}
    expected = sha256(data).hexdigest()
    for model, path in paths.items():
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["checkpoint"]["model"] != model or manifest["panel_sha256"] != expected:
            raise ValueError(f"Input coverage differs for {model}")
    per_model = {model: defaultdict(Counter) for model in paths}
    pooled = defaultdict(Counter)
    classes = {"right": defaultdict(Counter), "wrong": defaultdict(Counter)}
    covered = 0
    for start in range(0, min(len(cases), max_cases), 32):
        name = f"block-{start:06d}.json.gz"
        if any(not (path / name).is_file() for path in paths.values()):
            break
        batch = cases[start:min(len(cases), max_cases, start + 32)]
        for model, path in paths.items():
            with gzip.open(path / name, "rt") as file:
                rows = json.load(file)[:len(batch)]
            if [r["id"] for r in rows] != [c["id"] for c in batch]:
                raise ValueError(f"Mismatched episode panel in {path}/{name}")
            for case, row in zip(batch, rows, strict=True):
                if case["correct"] != row["correct"] or set(row["calls"]) != set(ARMS):
                    raise ValueError(f"Mismatched labels or calls for {case['id']}")
                update(per_model[model], row)
                update(pooled, row)
                stores = classes["right" if case["correct"] else "wrong"]
                stores[case["store"]]["count"] += 1
                receipt = row["calls"]["receipt"]["choice"] == "approve"
                for arm in ("card", "reread"):
                    stores[case["store"]][arm] += receipt - (row["calls"][arm]["choice"] == "approve")
        covered += len(batch)
    if not covered:
        raise ValueError("No shared audit cases have finished")
    supplementary = {}
    for path in sorted((root / "checkout-audit").glob("*/0/summary.json")):
        data = json.loads(path.read_text())
        model = data["model"]
        if model in MODELS or data["groups"]["all"]["tasks"] != len(cases) or covered != len(cases):
            continue
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest["checkpoint"]["model"] != model or manifest["panel_sha256"] != expected:
            raise ValueError(f"Supplementary panel differs for {model}")
        supplementary[model] = data["groups"]
    return {"revision": REVISION, "split": panel["split"], "coverage": panel["coverage"],
            "pricing": pricing(cases), "matched_cases": covered, "eligible_cases": len(cases),
            "models": per_model, "pooled": pooled, "supplementary_models": supplementary,
            "receipt_approval_deltas": {key: {arm: store_interval(stores, arm) for arm in ("card", "reread")}
                                        for key, stores in classes.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-cases", type=int, default=1_000_000)
    parser.add_argument("--split", choices=("test_cross_store", "test_cross_template"), default="test_cross_store")
    args = parser.parse_args()
    value = report(args.run_root, args.max_cases, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, value)


if __name__ == "__main__":
    main()
