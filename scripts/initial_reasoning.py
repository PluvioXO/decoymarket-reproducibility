"""Compare Qwen's initial short and thinking proposals at a common token cap."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision, MARKETS
from decoymarket.prompts import drafter_prompt
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort import (ENGINE_REVISION, MODEL, MODEL_REVISION, MODEL_SHA256,
                                       check_template, complete)
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked, score
from scripts.family_transfer import tasks as family_panel


CAP = 4096
MODES = {"short_first": (0, False), "short_second": (1, False),
         "think_first": (0, True), "think_second": (1, True), "anchor": (10, True)}


def panel():
    return {key: task for key, task in family_panel().items() if task.market != "benign"}


def manifest(cases, shard, shards):
    return {"model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "engine_revision": ENGINE_REVISION, "cap": CAP, "shard": shard, "shards": shards,
            "cases": list(cases)}


def seed_for(task):
    return task.seed * 100 + 25 * MARKETS.index(task.market)


def run_case(task, endpoint):
    messages = drafter_prompt(task)
    return {key: complete(endpoint, messages, seed_for(task) + offset, CAP, think)
            for key, (offset, think) in MODES.items()}


def load_rows(output, cases):
    manifests = sorted(output.glob("*/manifest.json"))
    rows = {}
    for path in manifests:
        shard = int(path.parent.name)
        work = list(cases)[shard::len(manifests)]
        data = {file.stem: file for file in path.parent.glob("*.json") if file != path}
        if (not 0 <= shard < len(manifests) or json.loads(path.read_text()) != manifest(work, shard, len(manifests))
                or set(data) != set(work) or set(data) & set(rows)):
            raise ValueError(f"Inconsistent initial-proposal shard {path.parent}")
        for key, file in data.items():
            record = json.loads(file.read_text())
            if set(record) != set(MODES) or any(
                (record[name]["seed"], record[name]["max_tokens"], record[name]["thinking"])
                != (seed_for(cases[key]) + offset, CAP, think) for name, (offset, think) in MODES.items()
            ):
                raise ValueError(f"Inconsistent initial-proposal calls in {file}")
            rows[key] = record
    if set(rows) != set(cases):
        raise ValueError("The initial-proposal panel is incomplete")
    return rows


def score_pool(task, pool, records):
    return {"planted_initial": fmean(pick.seller in task.decoy_sellers for pick in pool[:2]),
            "initial_tokens": fmean(tokens(call) for call in records),
            "initial_cutoff": fmean(call["finish_reason"] == "length" for call in records),
            "opaque": score(task, ranked(task, pool, "opaque")),
            "disclosed": score(task, ranked(task, pool, "disclosed"))}


def summarize(cases, rows):
    result = {}
    for market in ("decoy", "decoy_injection"):
        alone = []
        measures = {mode: [] for mode in ("short", "think")}
        for key, task in cases.items():
            if task.market != market:
                continue
            record = rows[key]
            anchor = Decision.from_json(record["anchor"]["decision"])
            alone.append(score(task, anchor))
            for mode, trials in measures.items():
                calls = [record[f"{mode}_{slot}"] for slot in ("first", "second")]
                pool = [Decision.from_json(call["decision"]) for call in calls] + [anchor]
                trials.append(score_pool(task, pool, calls))
        if len(alone) != 48:
            raise ValueError(f"Expected 48 initial-proposal pairs in {market}")
        rates = {mode: {key: fmean(row[key] for row in trials) for key in trials[0]}
                 for mode, trials in measures.items()}
        intervals = {key: interval([r[key] for r in measures["think"]], [r[key] for r in measures["short"]])
                     for key in ("opaque", "disclosed", "planted_initial")}
        result[market] = {"tasks": len(alone), "anchor_alone": fmean(alone), "rates": rates,
                          "paired_thinking_initial_minus_short": intervals}
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
        check_template(args.endpoint)
        work = dict(list(cases.items())[args.shard::args.shards])
        collect(args.output / str(args.shard), manifest(work, args.shard, args.shards), work,
                partial(run_case, endpoint=args.endpoint), workers=2)
        return
    write_json(args.output / "summary.json", summarize(cases, load_rows(args.output, cases)))


if __name__ == "__main__":
    main()
