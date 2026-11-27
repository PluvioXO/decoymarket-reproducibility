"""Replay the opaque-check pick from 16 drafts with the other model as reviewer."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from decoymarket.analysis import classify, paired_delta_ci
from decoymarket.backends import LiveBackend
from decoymarket.benchmark import Decision, classify_failure, select, task_from_json
from decoymarket.run import RunConfig, _dec_json, _result_json, _review_one
from scripts.collect_live import write_json

MODELS = ("qwen3-8b", "llama31-8b")
OUTPUT = Path("results/cross_model_review")
SEEDS = tuple(range(10001, 10201))
REGIME = "opaque"


def load_source(drafter, split="test"):
    seeds = SEEDS if split == "test" else tuple(range(1, 101))
    path = Path("results") / drafter / f"decoy_{split}.json"
    data = path.read_bytes()
    episode = json.loads(data)
    if (
        episode["backend"] != "live"
        or episode["is_mock"]
        or episode["stopped_early"]
        or episode["model_name"] != drafter
        or episode["market"] != "decoy"
        or tuple(row["task"]["seed"] for row in episode["tasks"]) != seeds
        or any(len(row["drafts"]) != 16 for row in episode["tasks"])
    ):
        raise ValueError(f"Expected {len(seeds)} full live decoy pools in {path}")
    return episode, hashlib.sha256(data).hexdigest()


def review_inputs(row):
    task = task_from_json(row["task"])
    drafts = [Decision.from_json(record["decision"]) for record in row["drafts"]]
    index = select(task, drafts, REGIME)
    key = row["drafts"][index]["key"]
    # The original collector built the review prompt from the first draft with this key.
    selected = next(d for d, record in zip(drafts, row["drafts"]) if record["key"] == key)
    return task, selected, row["reviews"][REGIME][key]


def collect(args):
    source, source_hash = load_source(args.drafter)
    reviewer_source, _ = load_source(args.reviewer)
    if (
        args.model_id != reviewer_source["model_id"]
        or args.revision != reviewer_source["collection"]["weights_revision"]
        or any(a["task"] != b["task"] for a, b in zip(source["tasks"], reviewer_source["tasks"]))
    ):
        raise ValueError("Expected the original reviewer weights and matched tasks")
    name = f"{args.drafter}_to_{args.reviewer}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUTPUT / f"{name}.tasks"
    checkpoints.mkdir(exist_ok=True)
    manifest_path = checkpoints / "manifest.json"
    manifest = {
        "schema": "decoymarket.cross_model_review.v1",
        "drafter": args.drafter,
        "reviewer": args.reviewer,
        "reviewer_model_id": args.model_id,
        "reviewer_weights_revision": args.revision,
        "source_sha256": source_hash,
        "market": "decoy",
        "regime": REGIME,
        "n_drafts": 16,
        "max_completion_tokens": source["collection"]["max_tokens"],
        "workers": 16,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Checkpoint settings changed for {name}")
    else:
        write_json(manifest_path, manifest)

    backend = LiveBackend(args.endpoint, args.model_id)
    config = RunConfig(
        model_name=args.reviewer, market="decoy", seeds=[], backend="live",
        max_tokens=manifest["max_completion_tokens"], run_seed=source["run_seed"],
    )
    rows = dict(zip(SEEDS, source["tasks"]))

    def review(seed):
        task, selected, same = review_inputs(rows[seed])
        index = same["seed"] - config.run_seed * 1000 - 500
        cross = _review_one(config, backend, None, task, selected, REGIME, None, index)
        if cross["prompt_tokens"] <= 0 or cross["completion_tokens"] < 0:
            raise ValueError(f"Missing token counts on task {seed}")
        outcome = {
            role: classify_failure(task, decision)["success"]
            for role, decision in (
                ("same", Decision.from_json(same["decision"])),
                ("cross", cross["decision"]),
            )
        }
        write_json(checkpoints / f"{seed}.json", {
            "task_seed": seed,
            "selection_category": classify(task, selected, REGIME),
            "selected": _dec_json(selected),
            "same_review": same,
            "cross_review": _result_json(cross),
            "optimal": outcome,
        })

    missing = [seed for seed in SEEDS if not (checkpoints / f"{seed}.json").exists()]
    done = len(SEEDS) - len(missing)
    print(f"{name}: {done}/{len(SEEDS)} reviews saved", flush=True)
    with ThreadPoolExecutor(max_workers=manifest["workers"]) as executor:
        futures = [executor.submit(review, seed) for seed in missing]
        for done, future in enumerate(as_completed(futures), done + 1):
            future.result()
            if done == 1 or done % 25 == 0 or done == len(SEEDS):
                print(f"{name}: {done}/{len(SEEDS)} reviews saved", flush=True)
    results = [json.loads((checkpoints / f"{seed}.json").read_text()) for seed in SEEDS]
    write_json(OUTPUT / f"{name}.json", {"collection": manifest, "tasks": results})


def summarize_direction(episode):
    rows = episode["tasks"]
    if tuple(row["task_seed"] for row in rows) != SEEDS:
        raise ValueError("Expected one paired review per decoy test task")
    same = [row["optimal"]["same"] for row in rows]
    cross = [row["optimal"]["cross"] for row in rows]
    delta, lo, hi = paired_delta_ci(cross, same)
    decoys = [row for row in rows if row["selection_category"] == "A"]
    optimal = [row for row in rows if row["selection_category"] == "S"]
    return {
        "tasks": len(rows),
        "same_model_optimal": sum(same),
        "cross_model_optimal": sum(cross),
        "cross_minus_same": delta,
        "paired_bootstrap_95_ci": [lo, hi],
        "category_a_choices": len(decoys),
        "category_a_repairs": {
            role: sum(row["optimal"][role] for row in decoys) for role in ("same", "cross")
        },
        "originally_optimal_choices": len(optimal),
        "optimal_choices_broken": {
            role: sum(not row["optimal"][role] for row in optimal) for role in ("same", "cross")
        },
        "mean_review_tokens": {
            role: mean(row[f"{role}_review"]["prompt_tokens"] + row[f"{role}_review"]["completion_tokens"]
                       for row in rows) for role in ("same", "cross")
        },
        "cross_review_tokens": sum(
            row["cross_review"]["prompt_tokens"] + row["cross_review"]["completion_tokens"]
            for row in rows
        ),
    }


def summarize():
    summary = {}
    for drafter, reviewer in (MODELS, MODELS[::-1]):
        name = f"{drafter}_to_{reviewer}"
        episode = json.loads((OUTPUT / f"{name}.json").read_text())
        summary[name] = summarize_direction(episode)
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("review")
    run.add_argument("drafter", choices=MODELS)
    run.add_argument("reviewer", choices=MODELS)
    for field in ("model-id", "endpoint", "revision"):
        run.add_argument(f"--{field}", required=True)
    actions.add_parser("summarize")
    args = parser.parse_args()
    if args.action == "summarize":
        summarize()
    elif args.drafter == args.reviewer:
        parser.error("choose the other model as reviewer")
    else:
        collect(args)


if __name__ == "__main__":
    main()
