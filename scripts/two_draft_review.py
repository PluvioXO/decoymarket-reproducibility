"""Collect Qwen reviews of Llama's possible two-draft picks in standard DecoyMarket."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from decoymarket.backends import LiveBackend
from decoymarket.benchmark import Decision, selection_key, task_from_json
from decoymarket.run import RunConfig, _result_json, _review_one
from scripts.auto_effort import ACTIONS, REGIME, replay_task
from scripts.collect_live import write_json
from scripts.cross_model_review import load_source, review_inputs
from scripts.effort_extensions import calibrated_policies, collect, evaluate_policies

DRAFTER = "llama31-8b"
REVIEWER = "qwen3-8b"
OUTPUT = Path("results/two_draft_review")
SPLITS = ("cal", "test")
MENU = (*ACTIONS, "cross_review")
TOKEN_PRICE = 10 / 1_000_000


def plan():
    sources = {split: load_source(DRAFTER, split) for split in SPLITS}
    qwen, qwen_hash = load_source(REVIEWER)
    previous_path = Path("results/cross_model_review/llama31-8b_to_qwen3-8b.json")
    previous_bytes = previous_path.read_bytes()
    previous = json.loads(previous_bytes)
    previous_info = previous["collection"]
    revision = qwen["collection"]["weights_revision"]
    if (
        previous_info["source_sha256"] != sources["test"][1]
        or previous_info["reviewer_model_id"] != qwen["model_id"]
        or previous_info["reviewer_weights_revision"] != revision
        or previous_info["regime"] != REGIME
        or previous_info["n_drafts"] != 16
    ):
        raise ValueError("The reusable reviews do not match the source pools and model")

    old = {}
    for row, review in zip(sources["test"][0]["tasks"], previous["tasks"], strict=True):
        task, selected, same = review_inputs(row)
        if (
            task.seed != review["task_seed"]
            or selected != Decision.from_json(review["selected"])
            or same != review["same_review"]
            or same["seed"] != review["cross_review"]["seed"]
        ):
            raise ValueError(f"Reusable review disagrees with the source task {task.seed}")
        old[task.seed] = (selected, same["seed"], review["cross_review"])

    cases, reused, rows, counts = {}, [], {}, {}
    for split, (episode, _) in sources.items():
        config = RunConfig(REVIEWER, "decoy", [], backend="live", run_seed=episode["run_seed"],
                           max_tokens=episode["collection"]["max_tokens"])
        rows[split] = episode["tasks"]
        before = len(cases)
        for row in episode["tasks"]:
            task = task_from_json(row["task"])
            drafts = row["drafts"]
            decisions = [Decision.from_json(draft["decision"]) for draft in drafts]
            order = sorted(range(len(drafts)), key=lambda i: selection_key(task, decisions[i], REGIME) + (i,))
            first_by_key = {}
            for draft, decision in zip(drafts, decisions):
                first_by_key.setdefault(draft["key"], decision)
            # Every ranked draft except the last can win a pair. Repeated keys share a review.
            for key in dict.fromkeys(drafts[i]["key"] for i in order[:-1]):
                selected = first_by_key[key]
                seed = row["reviews"][REGIME][key]["seed"]
                record = {"task_seed": task.seed, "draft_key": key, "seed": seed}
                prior = old.get(task.seed)
                if prior and (selected, seed) == prior[:2]:
                    reused.append({**record, "origin": "sixteen_draft_review", "review": prior[2]})
                else:
                    cases[f"{task.seed}-{seed}"] = (record, config, task, selected)
        counts[split] = len(cases) - before
    if len(reused) != len(old):
        raise ValueError("Expected one reusable review for each test task")
    manifest = {
        "schema": "decoymarket.two_draft_review.v1",
        "market": "decoy",
        "regime": REGIME,
        "drafter": DRAFTER,
        "reviewer": REVIEWER,
        "reviewer_model_id": qwen["model_id"],
        "reviewer_weights_revision": revision,
        "llama_source_sha256": {split: sources[split][1] for split in SPLITS},
        "qwen_reference_sha256": qwen_hash,
        "sixteen_draft_reviews_sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "new_calls_by_split": counts,
        "reused_calls": len(reused),
    }
    return manifest, cases, reused, rows


def collect_reviews(args, manifest, cases, reused):
    if args.revision != manifest["reviewer_weights_revision"]:
        raise ValueError("Expected the Qwen revision used by the primary study")
    backend = LiveBackend(args.endpoint, manifest["reviewer_model_id"])

    def run(case):
        record, config, task, selected = case
        index = record["seed"] - config.run_seed * 1000 - 500
        result = _result_json(_review_one(config, backend, None, task, selected, REGIME, None, index))
        if result["seed"] != record["seed"] or result["prompt_tokens"] <= 0:
            raise ValueError(f"Invalid request seed or token count for task {task.seed}")
        return {**record, "origin": "new", "review": result}

    new = collect(OUTPUT / "reviews.tasks", manifest, cases, run, args.workers)
    write_json(OUTPUT / "reviews.json", {"collection": manifest, "reviews": new + reused})
    print(f"Collected {len(new)} new reviews and reused {len(reused)}", flush=True)


def summarize(manifest, cases, reused, rows):
    raw = (OUTPUT / "reviews.json").read_bytes()
    dataset = json.loads(raw)
    planned = [case[0] for case in cases.values()] + reused
    expected = {(record["task_seed"], record["draft_key"]): record["seed"] for record in planned}
    seen, reviews = set(), defaultdict(dict)
    if dataset["collection"] != manifest:
        raise ValueError("Review logs do not match the source manifest")
    for record in dataset["reviews"]:
        seed, key, review = record["task_seed"], record["draft_key"], record["review"]
        identity = (seed, key)
        if identity in seen or expected.get(identity) != review["seed"] or review["prompt_tokens"] <= 0:
            raise ValueError(f"Unexpected review for {seed} and {key}")
        reviews[seed][key] = review
        seen.add(identity)
    if seen != expected.keys():
        raise ValueError("Some two-draft reviews are missing")

    scenarios = {}
    for penalty in (0.20, 0.10, 0.50):
        estimates = {
            split: [list(replay_task(row, penalty, reviews[row["task"]["seed"]])) for row in rows[split]]
            for split in SPLITS
        }
        policies, fit = calibrated_policies(estimates["cal"], MENU, TOKEN_PRICE)
        policies["agree2"] = lambda pair: "stop" if pair.state == "agreement" else "review"
        comparisons = (("cross_review", "review"), ("cross_review", "draft"), ("cross_review", "stop"),
                       ("auto", "best_fixed"), ("auto", "budget_only"), ("auto", "agree2"))
        report = evaluate_policies(estimates["test"], policies, MENU, TOKEN_PRICE, comparisons)
        scenarios[str(penalty)] = {**fit, **report}
    write_json(OUTPUT / "summary.json", {
        "collection": manifest,
        "review_log_sha256": hashlib.sha256(raw).hexdigest(),
        "initial_drafts": 2,
        "calibration_tasks": len(rows["cal"]),
        "dollars_per_million_tokens": 10,
        "scenarios_by_violation_penalty": scenarios,
    })
    print(f"Saved {OUTPUT / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "summarize"))
    parser.add_argument("--endpoint")
    parser.add_argument("--revision")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "collect" and (not args.endpoint or not args.revision or args.workers < 1):
        parser.error("collect requires an endpoint, a revision, and a positive worker count")
    manifest, cases, reused, rows = plan()
    if args.stage == "collect":
        collect_reviews(args, manifest, cases, reused)
    else:
        summarize(manifest, cases, reused, rows)


if __name__ == "__main__":
    main()
