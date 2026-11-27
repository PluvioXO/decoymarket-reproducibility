"""Replay a common short-action menu on held-out tasks across model families."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from decoymarket.benchmark import Decision, generate_task
from decoymarket.prompts import drafter_prompt
from scripts.calibration_sensitivity import nested_replay
from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS, MODEL, cases_for_block, check_template, complete, load_checkpoint
from scripts.draft_family_expansion import CAP, manifest, read_shards, seed_for, verify_template, with_encoding
from scripts.effort_baselines import compare
from scripts.effort_extensions import collect
from scripts.effort_followup import load_panel, review_messages
from scripts.effort_followup_report import decision_point, read_responses, replay
from scripts.family_transfer import MODEL as GPT_OSS, verify_template as verify_reasoning
from scripts.model_breadth import caps
from scripts.model_breadth_report import MODELS


BLOCKS = range(6, 24)
DRAFTS = 3
MISTRAL_REVIEW_CAP = 512
MENU = ("stop", "draft", "review")
ARCHIVED_QWEN = (MODEL, "qwen3-4b-q4", "qwen3-14b-q4")
REGIMES = ("opaque", "disclosed")
SCENARIOS = ((.2, 10, 0), (.5, 10, 0))


def panel():
    cases = {}
    for block in BLOCKS:
        for market, seed, split in cases_for_block(block):
            task = generate_task(seed, market)
            cases[task.task_id] = task, split
    return cases


def run_case(case, endpoint, model, encoding, reference=None):
    task, _ = case
    seed = seed_for(task)
    thinking = model == GPT_OSS
    request = {"template_kwargs": {"reasoning_effort": "low"}} if thinking else {}
    draft_cap = caps(model)[0]

    def call(messages, offset, cap):
        return complete(endpoint, with_encoding(messages, encoding), seed + offset, cap, thinking, model, **request)

    if reference is None:
        messages = drafter_prompt(task)
        short = [call(messages, offset, draft_cap) for offset in range(DRAFTS)]
        cap = draft_cap
    else:
        short = reference[task.task_id]["short"]
        cap = MISTRAL_REVIEW_CAP
    drafts = [Decision.from_json(call["decision"]) for call in short]
    opaque, disclosed = (review_messages(task, drafts, regime) for regime in REGIMES)
    row = {"review": call(opaque, 8, cap)}
    if disclosed != opaque:
        row["disclosed_review"] = call(disclosed, 8, cap)
    if reference is None:
        row["short"] = short
    return row


def saved_qwen(source, checkpoint, followup, cases):
    plan, entries = load_panel(source)
    if any(plan[key] != checkpoint[key] for key in CHECKPOINT_KEYS):
        raise ValueError("Saved Qwen3 logs must match the pinned study")
    reviews = read_responses(followup, "disclosure", plan, entries)
    if reviews is None:
        raise ValueError("The Qwen3 regime-matched reviews are required")
    rows = {}
    for task, row, _, _ in entries:
        key = task.task_id
        if key not in cases:
            continue
        if cases[key] != (task, row["split"]):
            raise ValueError(f"Saved task differs from the specified panel: {key}")
        rows[key] = {"short": row["short"][:DRAFTS], "review": row["review"]}
        if key in reviews:
            rows[key]["disclosed_review"] = reviews[key]["review"]
    return rows


def validated_entry(task, row, *, review_cap=CAP, draft_cap=CAP, thinking=False):
    drafts = [Decision.from_json(call["decision"]) for call in row["short"]]
    if len(drafts) != DRAFTS:
        raise ValueError(f"Expected {DRAFTS} drafts on {task.task_id}")
    changes = review_messages(task, drafts, "opaque") != review_messages(task, drafts, "disclosed")
    if changes != ("disclosed_review" in row):
        raise ValueError(f"Missing or unexpected regime-matched review on {task.task_id}")
    seed = seed_for(task)
    requests = ([(seed + offset, draft_cap, thinking) for offset in range(DRAFTS)]
                + [(seed + 8, review_cap, thinking)] * (1 + changes))
    calls = [*row["short"], row["review"]] + ([row["disclosed_review"]] if changes else [])
    if [(call["seed"], call["max_tokens"], call["thinking"]) for call in calls] != requests:
        raise ValueError(f"Unexpected call settings on {task.task_id}")
    return task, row, drafts, {}


def summarize(cases, rows, *, review_cap=CAP, draft_cap=CAP, thinking=False):
    if cases.keys() != rows.keys():
        raise ValueError("Saved results must cover the specified panel")
    points = defaultdict(list)
    diagnostics = Counter()
    for key, (task, split) in cases.items():
        row = rows[key]
        entry = validated_entry(task, row, review_cap=review_cap, draft_cap=draft_cap, thinking=thinking)
        disclosed = row.get("disclosed_review")
        points["opaque", split].append(decision_point(entry, "opaque", None))
        points["disclosed", split].append(decision_point(entry, "disclosed", {key: {"review": disclosed}} if disclosed else {}))
        diagnostics["new_disclosed_reviews"] += disclosed is not None
        if split == "test":
            diagnostics["draft_cutoffs"] += sum(call["finish_reason"] == "length" for call in row["short"])
            diagnostics["opaque_review_cutoffs"] += row["review"]["finish_reason"] == "length"
            diagnostics["disclosed_review_cutoffs"] += (disclosed or row["review"])["finish_reason"] == "length"
    result = {}
    for regime in REGIMES:
        cal, test = (points[regime, split] for split in ("cal", "test"))
        estimates = replay(cal, test, MENU, SCENARIOS)
        for row in estimates:
            penalty = row["penalty"]
            row["heuristics"] = compare(cal, test, penalty, menu=MENU)
            row["refitted"] = nested_replay(cal, test, penalty, menu=MENU)
        result[regime] = estimates
    return {"blocks": list(BLOCKS), "calibration_tasks": len(points["opaque", "cal"]),
            "test_tasks": len(points["opaque", "test"]), "diagnostics": diagnostics, "regimes": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--qwen-disclosure", type=Path)
    parser.add_argument("--review-reference", type=Path)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint["model"]
    if model not in MODELS:
        parser.error("checkpoint is outside the matched protocol")
    if args.qwen_disclosure and (args.endpoint or args.review_reference or model not in ARCHIVED_QWEN):
        parser.error("archived disclosures apply to an offline Qwen3 replay")
    cases = panel()
    reference = None
    draft_cap = caps(model)[0]
    cap = draft_cap
    if args.review_reference:
        if checkpoint["model"] != "mistral-7b-v03-q4":
            parser.error("the 512-token review repeat uses Mistral")
        reference, reference_encoding = read_shards(args.review_reference, checkpoint, cases, drafts=DRAFTS)
        cap = MISTRAL_REVIEW_CAP
    if args.endpoint:
        if model == MODEL or not 0 <= args.shard < args.shards <= len(cases):
            parser.error("invalid collection settings")
        if model == GPT_OSS:
            verify_reasoning(args.endpoint)
            encoding = "native"
        else:
            if model in ARCHIVED_QWEN:
                check_template(args.endpoint)
            encoding = verify_template(args.endpoint, generate_task(900122, "decoy"))
        if reference is not None and encoding != reference_encoding:
            raise ValueError("The review template differs from the original collection")
        work = dict(list(cases.items())[args.shard::args.shards])
        collect(args.output / str(args.shard), manifest(checkpoint, work, args.shard, args.shards, encoding,
                drafts=DRAFTS, review_cap=cap, completion_cap=draft_cap), work, partial(
                run_case, endpoint=args.endpoint, model=model, encoding=encoding, reference=reference), workers=2)
        return
    if args.qwen_disclosure:
        rows, encoding = saved_qwen(args.checkpoint, checkpoint, args.qwen_disclosure, cases), "native"
    elif model == MODEL:
        parser.error("the recorded Qwen3-8B comparison needs the disclosure run")
    else:
        rows, encoding = read_shards(args.output, checkpoint, cases, drafts=DRAFTS, review_cap=cap, completion_cap=draft_cap)
    summary = {"checkpoint": checkpoint, "prompt_encoding": encoding}
    if reference is not None:
        if encoding != reference_encoding or any("short" in row for row in rows.values()):
            raise ValueError("The cap repeat must reuse the original drafts and encoding")
        rows = {key: {"short": reference[key]["short"], **row} for key, row in rows.items()}
        summary["review_cap"] = cap
    if model == GPT_OSS:
        summary.update({"draft_cap": draft_cap, "review_cap": cap, "reasoning_effort": "low"})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", {**summary, **summarize(
        cases, rows, review_cap=cap, draft_cap=draft_cap, thinking=model == GPT_OSS)})


if __name__ == "__main__":
    main()
