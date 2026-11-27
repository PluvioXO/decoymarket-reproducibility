"""Collect and score a held-out paired buyer–seller exchange."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import classify_failure, task_from_json
from scripts.collect_live import write_json
from scripts.distributed_effort import ENGINE_REVISION, MODEL, MODEL_REVISION, MODEL_SHA256, check_template
from scripts.distributed_effort_report import task_entry
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked
from scripts.seller_agent_pilot import run_case, summarize


SPLITS = (("cal", 231001, 48), ("test", 241001, 144))
ARMS = (("static", "opening_task"), ("adaptive", "final_task"))


def panel():
    return {f"{split}-{seed}": (split, seed) for split, start, count in SPLITS
            for seed in range(start, start + count)}


def manifest(cases, shard, shards):
    return {"model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "engine_revision": ENGINE_REVISION, "shard": shard, "shards": shards,
            "cases": list(cases)}


def load_rows(output):
    cases = panel()
    manifests = sorted(output.glob("*/manifest.json"))
    rows = {}
    for path in manifests:
        shard = int(path.parent.name)
        expected = dict(list(cases.items())[shard::len(manifests)])
        files = {p.stem: p for p in path.parent.glob("*.json") if p != path}
        if (not 0 <= shard < len(manifests) or json.loads(path.read_text()) != manifest(expected, shard, len(manifests))
                or set(files) != set(expected) or set(files) & set(rows)):
            raise ValueError(f"Incomplete or inconsistent exchange shard {path.parent}")
        for key, location in files.items():
            row = json.loads(location.read_text())
            if (row["split"], row["seed"]) != expected[key]:
                raise ValueError(f"Unexpected exchange case {location}")
            if row["opening_task"] == row["final_task"] and row["static"] != row["adaptive"]:
                raise ValueError(f"The unchanged offer set must reuse the same responses in {location}")
            rows[key] = row
    if set(rows) != set(cases):
        raise ValueError("The held-out exchange is incomplete")
    return [rows[key] for key in cases]


def candidate_outcomes(row, arm, task_key):
    task, _, drafts, thinking = task_entry(task_from_json(row[task_key]), row[arm])
    thought = thinking[2048]
    pool = [*drafts[:2], thought]
    decisions = {"thinking": thought, "opaque": ranked(task, pool, "opaque"),
                 "disclosed": ranked(task, pool, "disclosed")}
    outcomes = {}
    for name, choice in decisions.items():
        result = classify_failure(task, choice)
        outcomes[name] = {"optimal": result["success"], "failure": result["violation"] not in ("none", "overpay")}
    return outcomes


def paired_outcomes(rows):
    static, adaptive = [[candidate_outcomes(row, arm, key) for row in rows] for arm, key in ARMS]
    result = {}
    for mode, fields in static[0].items():
        result[mode] = {}
        for field in fields:
            before = [r[mode][field] for r in static]
            after = [r[mode][field] for r in adaptive]
            result[mode][field] = {"static": fmean(before), "adaptive": fmean(after),
                                   "paired_adaptive_minus_static": interval(after, before)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    cases = panel()
    if args.endpoint:
        if not 0 <= args.shard < args.shards <= len(cases):
            parser.error("invalid shard")
        if os.environ.get("PYTHONHASHSEED") != "0":
            parser.error("set PYTHONHASHSEED=0 to reproduce the market seeds")
        check_template(args.endpoint)
        args.buyer_endpoint = args.seller_endpoint = args.endpoint
        args.buyer_model = args.seller_model = MODEL
        work = dict(list(cases.items())[args.shard::args.shards])
        collect(args.output / str(args.shard), manifest(work, args.shard, args.shards), work,
                lambda case: run_case(args, case), workers=2)
        return
    rows = load_rows(args.output)
    design = {"model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
              "engine_revision": ENGINE_REVISION, "splits": SPLITS, "cases": list(cases),
              "shards": sum(1 for _ in args.output.glob("*/manifest.json"))}
    summary = summarize(rows, design)
    tests = [row for row in rows if row["split"] == "test"]
    summary["paired_candidate_outcomes"] = paired_outcomes(tests)
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
