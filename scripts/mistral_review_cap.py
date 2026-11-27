"""Repeat the matched Mistral reviews at a higher completion cap."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import MARKETS
from scripts.collect_live import write_json
from scripts.distributed_effort import complete, load_checkpoint
from scripts.draft_family_expansion import (first_pair_review_prompt, manifest, measure, read_shards,
                                            seed_for, tasks, verify_template)
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval


MODEL, REVIEW_CAP, SHARDS = "mistral-7b-v03-q4", 512, 4


def cap_manifest(checkpoint, prompts, shard, encoding):
    digest = sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return {**manifest(checkpoint, prompts, shard, SHARDS, encoding),
            "review_cap": REVIEW_CAP, "review_prompts_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["model"] != MODEL or not 0 <= args.shard < SHARDS:
        raise ValueError("The review-cap comparison requires Mistral and four shards")
    cases = tasks()
    rows, encoding = read_shards(args.source, checkpoint, cases)
    prompts = {key: first_pair_review_prompt(task, rows[key]["short"], encoding) for key, task in cases.items()}
    pairs = [dict(list(prompts.items())[shard::SHARDS]) for shard in range(SHARDS)]
    if args.endpoint:
        if verify_template(args.endpoint, next(iter(cases.values()))) != encoding:
            raise ValueError("The review template differs from the original study")
        work = {key: (cases[key], prompt) for key, prompt in pairs[args.shard].items()}

        def review(case):
            task, messages = case
            return complete(args.endpoint, messages, seed_for(task) + 8, REVIEW_CAP, False, MODEL)

        collect(args.output / str(args.shard), cap_manifest(checkpoint, pairs[args.shard], args.shard, encoding),
                work, review, workers=2)
        return

    reviews = {}
    for shard, work in enumerate(pairs):
        folder = args.output / str(shard)
        saved = json.loads((folder / "manifest.json").read_text())
        paths = {p.stem: p for p in folder.glob("*.json") if p.name != "manifest.json"}
        if saved != cap_manifest(checkpoint, work, shard, encoding) or paths.keys() != work.keys():
            raise ValueError(f"Unmatched review records in {folder}")
        reviews.update((key, json.loads(path.read_text())) for key, path in paths.items())
    results = {}
    for market in MARKETS:
        cases_in_market = [(key, task) for key, task in cases.items() if task.market == market]
        old = [measure(task, rows[key]) for key, task in cases_in_market]
        new = [measure(task, {**rows[key], "review": reviews[key]}, review_cap=REVIEW_CAP)
               for key, task in cases_in_market]
        before = [row["first_pair"] for row in old]
        at_256 = [row["first_pair_reviewed"] for row in old]
        at_512 = [row["first_pair_reviewed"] for row in new]
        results[market] = {"tasks": len(before), "initial": fmean(before),
                           "optimality": {"256": fmean(at_256), "512": fmean(at_512)},
                           "cutoff": {"256": fmean(row["review_cutoff"] for row in old),
                                      "512": fmean(row["review_cutoff"] for row in new)},
                           "parse": {"256": fmean(row["review_parse"] for row in old),
                                     "512": fmean(row["review_parse"] for row in new)},
                           "paired_512_vs_initial": interval(at_512, before),
                           "paired_512_vs_256": interval(at_512, at_256)}
    write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
