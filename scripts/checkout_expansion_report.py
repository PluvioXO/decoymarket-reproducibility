"""Summarize the complete task blocks shared by every checkout-replay model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

from scripts.checkout_expansion import BLOCK, SIMULATIONS, add_row, blocks, catalog_tasks
from scripts.collect_live import write_json


MODELS = (
    "aya8b-q4", "falcon3-7b-q4", "gemma-3-4b-q4", "gemma3-12b-q4", "gpt-oss-20b-mxfp4", "granite33-8b-q4",
    "llama31-8b-q4", "mistral-7b-v03-q4", "olmo3-7b-q4", "phi-4-mini-q8", "qwen3-14b-q4", "qwen3-4b-q4",
    "qwen3-8b-q4", "smollm3-3b-q4",
)
CONTRASTS = ("draft", "review", "visible_then_verify")


def summaries(root):
    available = {}
    for model in MODELS:
        shards = sorted((root / "checkout-expansion" / model).glob("*/manifest.json"))
        manifests = [json.loads(path.read_text()) for path in shards]
        if not manifests:
            raise ValueError(f"Missing {model} collection")
        if {m["shard"] for m in manifests} != set(range(manifests[0]["shards"])):
            raise ValueError(f"Incomplete shard manifest coverage for {model}")
        if any(m["checkpoint"]["model"] != model or m["shards"] != len(shards) for m in manifests):
            raise ValueError(f"Mismatched checkpoint or sharding for {model}")
        available[model] = {spec["shard"]: path.parent for path, spec in zip(shards, manifests, strict=True)}
    return available


def paired_interval(differences):
    if not differences:
        return None
    mean = fmean(differences)
    half = 1.96 * stdev(differences) / sqrt(len(differences)) if len(differences) > 1 else 0
    return {"delta_percentage_points": 100 * mean, "approx_95_percent_ci_points": [100 * (mean - half), 100 * (mean + half)],
            "unique_tasks": len(differences)}


def read_shared(root, max_blocks):
    archive = sorted(catalog_tasks(root / "catalog.json"), key=lambda task: sha256(task["id"].encode()).digest())
    available = summaries(root)
    report = {model: defaultdict(Counter) for model in available}
    prefixes = {model: {i: blocks(archive, i, len(paths)) for i in paths} for model, paths in available.items()}
    paired = defaultdict(list)
    done = 0
    for offset in range(0, min(SIMULATIONS, max_blocks * BLOCK), BLOCK):
        name = f"block-{offset:06d}.json.gz"
        if any(not (path / name).is_file() for paths in available.values() for path in paths.values()):
            break
        cases = defaultdict(list)
        for model, paths in available.items():
            for index, path in paths.items():
                expected_offset, expected = next(prefixes[model][index])
                with gzip.open(path / name, "rt") as handle:
                    rows = json.load(handle)
                if offset != expected_offset or [r["id"] for r in rows] != [task["id"] for task in expected]:
                    raise ValueError(f"Unexpected evaluation coverage in {path}/{name}")
                for row in rows:
                    add_row(report[model], row)
                    cases[row["id"]].append(row)
        for rows in cases.values():
            if len(rows) != len(available):
                raise ValueError(f"Partial model matching for {rows[0]['id']}")
            if rows[0]["domain"] != "simulated_checkout":
                continue
            for alternative in CONTRASTS:
                paired[alternative].append(fmean(int(row["score"]["source"][1]) - row["score"][alternative][1] for row in rows))
        done += 1
    if not done:
        raise ValueError("No completed matched blocks")
    return {"matched_blocks": done, "models": report,
            "simulated_paired_source_vs": {alternative: paired_interval(paired[alternative]) for alternative in CONTRASTS}}


def collect_evidence(directory):
    reports = {}
    for model in MODELS:
        path = directory / model / "0/summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            if summary["model"] != model:
                raise ValueError(f"Unexpected checkpoint for {path}")
            reports[model] = summary
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-blocks", type=int, default=SIMULATIONS // BLOCK)
    args = parser.parse_args()
    result = read_shared(args.run_root, args.max_blocks)
    result["evidence"] = collect_evidence(args.run_root / "evidence/checkout-evidence")
    result["updated_storefronts"] = collect_evidence(args.run_root / "retail/checkout-retail-updated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
