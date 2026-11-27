"""Collect only the new calls needed for the matched effort follow-up."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path

from decoymarket.benchmark import evaluate, select
from decoymarket.prompts import drafter_prompt, reviewer_prompt
from scripts.distributed_effort import ENGINE_REVISION, SHORT_DRAFTS, THINK_CAPS, cases_for_block, check_template, complete
from scripts.distributed_effort_report import load_blocks, task_entries
from scripts.effort_extensions import collect

PANEL_BLOCKS = 104
NOTE_BLOCKS = 20
CAP_BLOCKS = 4
LONG_MENU_BLOCKS = 16
EXPERIMENTS = ("disclosure", "note", "caps", "long-menu")


def load_panel(source):
    plan = json.loads((source / "plan.json").read_text())
    if (plan["engine_revision"], plan["short_drafts"], plan["think_caps"]) != (
        ENGINE_REVISION, SHORT_DRAFTS, list(THINK_CAPS),
    ):
        raise ValueError(f"Incompatible matched settings in {source}")
    rows, _ = load_blocks(source, plan)
    expected = {case for block in range(PANEL_BLOCKS) for case in cases_for_block(block)}
    last_seed = max(seed for _, seed, _ in expected)
    panel = [row for row in rows if row["seed"] <= last_seed]
    actual = [(row["market"], row["seed"], row["split"]) for row in panel]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"Missing or duplicate tasks in the 104-block panel in {source}")
    panel.sort(key=lambda row: (row["seed"], row["market"]))
    return plan, list(task_entries(panel))


def review_messages(task, drafts, regime):
    first = drafts[:2]
    pick = first[select(task, first, regime)]
    check = evaluate(task, pick, regime)
    report = {"seller": pick.seller, **{key: check[key] for key in ("passes", "failed", "visible")}}
    return reviewer_prompt(task, pick, "report", report)


def note_control(task):
    if sum(bool(offer.note) for offer in task.offers) != 1:
        raise ValueError(f"Expected one seller note on {task.task_id}")
    return replace(task, offers=[replace(offer, note="") if offer.note else offer for offer in task.offers])


def paired_cap_calls(task, row):
    messages = drafter_prompt(task)
    seed = row["thinking"]["2048"]["seed"]
    return {str(cap): (messages, seed, cap, True) for cap in (2048, 4096)}


def inputs(entry, experiment):
    task, row, drafts, _ = entry
    if experiment == "disclosure":
        messages = review_messages(task, drafts, "disclosed")
        if messages != review_messages(task, drafts, "opaque"):
            saved = row["review"]
            return {"review": (messages, saved["seed"], saved["max_tokens"], saved["thinking"])}
        return {}
    if experiment == "long-menu":
        return paired_cap_calls(task, row) if task.seed <= 100000 + 10 * LONG_MENU_BLOCKS else {}
    if row["split"] != "test":
        return {}
    if experiment == "note":
        if task.market != "decoy_injection" or task.seed > 100000 + 10 * NOTE_BLOCKS:
            return {}
        messages = drafter_prompt(note_control(task))
        saved = {"first": row["short"][0], "second": row["short"][1], "thinking": row["thinking"]["2048"]}
        return {name: (messages, record["seed"], record["max_tokens"], record["thinking"])
                for name, record in saved.items()}
    if experiment == "caps" and task.seed <= 100000 + 10 * CAP_BLOCKS:
        return paired_cap_calls(task, row)
    return {}


def jobs(entries, experiment):
    return {entry[0].task_id: calls for entry in entries if (calls := inputs(entry, experiment))}


def run_case(calls, model, endpoint):
    return {name: complete(endpoint, *args, model=model) for name, args in calls.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be between zero and shards minus one")
    plan, entries = load_panel(args.source)
    if args.experiment in ("note", "long-menu") and plan["model"] != "qwen3-8b-q4":
        parser.error(f"the {args.experiment} protocol uses Qwen3-8B")
    all_cases = jobs(entries, args.experiment)
    cases = dict(list(all_cases.items())[args.shard::args.shards])
    if not cases:
        parser.error("empty shard")
    manifest = {key: plan[key] for key in ("model", "model_revision", "model_sha256", "engine_revision")}
    manifest.update({
        "experiment": args.experiment, "shard": args.shard, "shards": args.shards,
        "calls_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
    })
    check_template(args.endpoint)
    run = partial(run_case, model=plan["model"], endpoint=args.endpoint)
    collect(args.output, manifest, cases, run, workers=1 if args.experiment == "long-menu" else 2)


if __name__ == "__main__":
    main()
