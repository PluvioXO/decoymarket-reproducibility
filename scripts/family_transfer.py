"""Collect an independent reasoning-family replication of the selector contrast."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision, MARKETS, generate_task
from decoymarket.prompts import drafter_prompt
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort import ENGINE_REVISION, cases_for_block, complete, post
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked, score

MODEL = "gpt-oss-20b-mxfp4"
REVISION = "41e8a94d15d62070b36cf4553a0250717ecf46b8"
WEIGHTS_SHA256 = "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"
CAP = 4096
ARMS = {"first": "low", "second": "low", "high": "high"}


def tasks():
    return {f"{market}-{seed}": generate_task(seed, market)
            for block in range(6) for market, seed, split in cases_for_block(block) if split == "test"}


def manifest(keys, shard, shards):
    return {"model": MODEL, "model_revision": REVISION, "model_sha256": WEIGHTS_SHA256,
            "engine_revision": ENGINE_REVISION, "shard": shard, "shards": shards,
            "cases": list(keys), "completion_cap": CAP, "reasoning_effort": ARMS}


def verify_template(endpoint):
    messages = drafter_prompt(generate_task(900001, "benign"))
    for effort in ("low", "high"):
        prompt = post(endpoint, "/apply-template", {
            "messages": messages, "chat_template_kwargs": {"reasoning_effort": effort},
        })["prompt"]
        if f"reasoning: {effort}" not in prompt.lower():
            raise ValueError(f"The serving template did not honor reasoning_effort={effort}")


def run_case(task, endpoint):
    messages = drafter_prompt(task)
    seed = task.seed * 100 + 25 * MARKETS.index(task.market)
    return {name: complete(endpoint, messages, seed + offset, CAP, True, MODEL,
                           template_kwargs={"reasoning_effort": effort})
            for offset, (name, effort) in enumerate(ARMS.items())}


def summarize(output, cases):
    manifests = sorted(output.glob("*/manifest.json"))
    results = {}
    for path in manifests:
        actual = json.loads(path.read_text())
        shard = actual["shard"]
        keys = list(cases)[shard::len(manifests)]
        if actual != manifest(keys, shard, len(manifests)) or set(keys) & set(results):
            raise ValueError(f"Invalid or duplicated shard {path}")
        for key in keys:
            record = json.loads((path.parent / f"{key}.json").read_text())
            if set(record) != set(ARMS) or any(call["max_tokens"] != CAP for call in record.values()):
                raise ValueError(f"Incomplete family-transfer call {key}")
            results[key] = record
    if set(results) != set(cases):
        raise ValueError("The reasoning-family replication is incomplete")

    panel = {}
    for market in MARKETS:
        high, opaque, disclosed, spent, cutoff, parsed = [], [], [], [], [], []
        low = []
        restored = 0
        for key, task in cases.items():
            if task.market != market:
                continue
            row = results[key]
            pool = [Decision.from_json(row[name]["decision"]) for name in ARMS]
            high.append(score(task, pool[-1]))
            opaque.append(score(task, ranked(task, pool, "opaque")))
            disclosed.append(score(task, ranked(task, pool, "disclosed")))
            restored += bool(high[-1] and not opaque[-1] and disclosed[-1])
            spent.append(tokens(row["high"]))
            cutoff.append(row["high"]["finish_reason"] == "length")
            parsed.append(row["high"]["parse_ok"])
            low.extend(row[name] for name, effort in ARMS.items() if effort == "low")
        panel[market] = {"tasks": len(high), "high_alone": fmean(high), "opaque": fmean(opaque),
                         "disclosed": fmean(disclosed), "disclosed_vs_opaque": interval(disclosed, opaque),
                         "opaque_vs_high": interval(opaque, high), "restored_optima": restored,
                         "high_tokens": fmean(spent), "high_cutoff": fmean(cutoff), "high_parse": fmean(parsed),
                         "low_tokens": fmean(tokens(record) for record in low),
                         "low_cutoff": fmean(record["finish_reason"] == "length" for record in low),
                         "low_parse": fmean(record["parse_ok"] for record in low)}
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    cases = tasks()
    if args.endpoint:
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        verify_template(args.endpoint)
        work = dict(list(cases.items())[args.shard::args.shards])
        spec = manifest(work, args.shard, args.shards)
        collect(args.output / str(args.shard), spec, work, partial(run_case, endpoint=args.endpoint), workers=2)
    else:
        write_json(args.output / "summary.json", summarize(args.output, cases))


if __name__ == "__main__":
    main()
