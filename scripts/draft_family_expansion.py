"""Collect and score the same short-draft protocol across model families."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import fmean
from urllib.error import HTTPError

from decoymarket.benchmark import Decision, MARKETS, classify_failure, evaluate
from decoymarket.estimators import no_review_curve
from decoymarket.prompts import drafter_prompt, reviewer_prompt
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS, ENGINE_REVISION, MODEL, complete, load_checkpoint, post
from scripts.distributed_effort_report import load_blocks
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked, score
from scripts.family_transfer import tasks


CAP, DRAFTS, BLOCKS = 256, 4, 6
ENCODINGS = ("native", "system-in-user")


def with_encoding(messages, encoding):
    if encoding == "native":
        return messages
    return [{"role": "user", "content": "\n\n".join(message["content"] for message in messages)}]


def verify_template(endpoint, task):
    messages = drafter_prompt(task)
    for encoding in ENCODINGS:
        try:
            rendered = post(endpoint, "/apply-template", {
                "messages": with_encoding(messages, encoding), "chat_template_kwargs": {"enable_thinking": False},
            })["prompt"]
        except HTTPError as error:
            if error.code not in (400, 500):
                raise
            continue
        if all(message["content"].strip() in rendered for message in messages):
            return encoding
    raise ValueError("The serving template did not preserve the shopper's messages")


def seed_for(task):
    return task.seed * 100 + 25 * MARKETS.index(task.market)


def first_pair_review_prompt(task, short, encoding):
    pick = ranked(task, [Decision.from_json(row["decision"]) for row in short[:2]], "opaque")
    check = evaluate(task, pick, "opaque")
    report = {"seller": pick.seller, **{key: check[key] for key in ("passes", "failed", "visible")}}
    return with_encoding(reviewer_prompt(task, pick, "report", report), encoding)


def run_case(task, endpoint, model, encoding):
    messages = with_encoding(drafter_prompt(task), encoding)
    seed = seed_for(task)
    short = [complete(endpoint, messages, seed + offset, CAP, False, model) for offset in range(DRAFTS)]
    review_messages = first_pair_review_prompt(task, short, encoding)
    return {"short": short, "review": complete(endpoint, review_messages, seed + 8, CAP, False, model)}


def manifest(checkpoint, keys, shard, shards, encoding, *, drafts=DRAFTS, review_cap=CAP, completion_cap=CAP):
    details = {**{key: checkpoint[key] for key in CHECKPOINT_KEYS}, "engine_revision": ENGINE_REVISION,
               "completion_cap": completion_cap, "drafts": drafts, "prompt_encoding": encoding,
               "shard": shard, "shards": shards, "cases": list(keys)}
    if review_cap != completion_cap:
        details["review_cap"] = review_cap
    return details


def read_shards(directory, checkpoint, cases, *, drafts=DRAFTS, review_cap=CAP, completion_cap=CAP):
    manifests = sorted(directory.glob("*/manifest.json"))
    rows, encodings = {}, set()
    for path in manifests:
        saved = json.loads(path.read_text())
        shard, encoding = saved["shard"], saved["prompt_encoding"]
        keys = list(cases)[shard::len(manifests)]
        if (not 0 <= shard < len(manifests) or encoding not in ENCODINGS
                or saved != manifest(checkpoint, keys, shard, len(manifests), encoding, drafts=drafts,
                                    review_cap=review_cap, completion_cap=completion_cap)
                or set(keys) & rows.keys()
                or {file.stem for file in path.parent.glob("*.json") if file != path} != set(keys)):
            raise ValueError(f"Invalid or incomplete shard in {path}")
        encodings.add(encoding)
        rows.update((key, json.loads((path.parent / f"{key}.json").read_text())) for key in keys)
    if rows.keys() != cases.keys() or len(encodings) != 1:
        raise ValueError("The panel is incomplete or mixes prompt encodings")
    return rows, encodings.pop()


def read_saved_qwen(directory, checkpoint):
    plan = json.loads((directory / "plan.json").read_text())
    if checkpoint["model"] != MODEL or any(plan[key] != checkpoint[key] for key in CHECKPOINT_KEYS):
        raise ValueError("Saved Qwen3-8B logs must match the pinned study")
    workers = {worker: {**spec, "blocks": min(spec["blocks"], BLOCKS - spec["first_block"])}
               for worker, spec in plan["workers"].items() if spec["first_block"] < BLOCKS}
    rows, _ = load_blocks(directory, {**plan, "workers": workers})
    return {f"{row['market']}-{row['seed']}": {"short": row["short"][:DRAFTS], "review": row["review"]}
            for row in rows if row["split"] == "test"}


def measure(task, row, *, review_cap=CAP):
    short, review = row["short"], row["review"]
    seed = seed_for(task)
    expected = [(seed + offset, CAP, False) for offset in range(DRAFTS)]
    expected.append((seed + 8, review_cap, False))
    calls = [*short, review]
    if [(call["seed"], call["max_tokens"], call["thinking"]) for call in calls] != expected:
        raise ValueError(f"Incompatible requests for {task.task_id}")
    drafts = [Decision.from_json(call["decision"]) for call in short]
    curve = no_review_curve(task, drafts, [tokens(call) for call in short], "opaque", (1, 2, 4))
    opaque, disclosed = (ranked(task, drafts, regime) for regime in ("opaque", "disclosed"))
    initial = ranked(task, drafts[:2], "opaque")
    revised = Decision.from_json(review["decision"]) if review["parse_ok"] else initial
    opaque_outcome, disclosed_outcome = (classify_failure(task, decision) for decision in (opaque, disclosed))
    return {
        **{f"opaque_{k}": row["success"] for k, row in curve.items()},
        "disclosed_4": disclosed_outcome["success"],
        "opaque_failure_4": opaque_outcome["violation"] not in ("none", "overpay"),
        "disclosed_failure_4": disclosed_outcome["violation"] not in ("none", "overpay"),
        "planted_displacement_restored": opaque.seller in task.decoy_sellers and disclosed_outcome["success"],
        "first_pair": score(task, initial), "first_pair_reviewed": score(task, revised),
        "draft_parse": fmean(call["parse_ok"] for call in short),
        "draft_cutoff": fmean(call["finish_reason"] == "length" for call in short),
        "review_parse": review["parse_ok"], "review_cutoff": review["finish_reason"] == "length",
        "draft_tokens": fmean(tokens(call) for call in short), "review_tokens": tokens(review),
    }


def summarize(cases, rows):
    if cases.keys() != rows.keys():
        raise ValueError("Saved results must cover the same task panel")
    markets = {}
    for market in MARKETS:
        measures = [measure(task, rows[key]) for key, task in cases.items() if task.market == market]
        if len(measures) != 48:
            raise ValueError(f"Expected 48 held-out tasks in {market}")
        values = {key: [item[key] for item in measures] for key in measures[0]}
        differences = {"scale_4_vs_1": ("opaque_4", "opaque_1"), "disclosure_at_4": ("disclosed_4", "opaque_4"),
                       "review_vs_first_pair": ("first_pair_reviewed", "first_pair")}
        markets[market] = {"tasks": len(measures), **{key: fmean(items) for key, items in values.items()},
                           "paired": {key: interval(values[a], values[b]) for key, (a, b) in differences.items()}}
    return markets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--saved-qwen", action="store_true")
    args = parser.parse_args()
    cases = tasks()
    checkpoint = load_checkpoint(args.checkpoint)
    if args.endpoint:
        if args.saved_qwen or not 0 <= args.shard < args.shards <= len(cases):
            parser.error("invalid collection settings")
        encoding = verify_template(args.endpoint, next(iter(cases.values())))
        work = dict(list(cases.items())[args.shard::args.shards])
        collect(args.output / str(args.shard), manifest(checkpoint, work, args.shard, args.shards, encoding),
                work, partial(run_case, endpoint=args.endpoint, model=checkpoint["model"], encoding=encoding), workers=2)
        return
    if args.saved_qwen:
        rows, encoding = read_saved_qwen(args.checkpoint, checkpoint), "native"
    else:
        rows, encoding = read_shards(args.output, checkpoint, cases)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", {"checkpoint": checkpoint, "prompt_encoding": encoding,
                                              "saved_qwen": args.saved_qwen, "markets": summarize(cases, rows)})


if __name__ == "__main__":
    main()
