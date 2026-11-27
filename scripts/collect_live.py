"""Collect a pilot or study cells with restartable task-level checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from decoymarket.benchmark import MARKETS, Decision, classify_failure, task_from_json
from decoymarket.run import RunConfig, run_cell


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data) + "\n")
    pending.replace(path)


def collect_cell(config, output, args):
    checkpoints = output.with_suffix(".tasks")
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_file = checkpoints / "manifest.json"
    manifest = {
        "model_id": config.model_id,
        "market": config.market,
        "seeds": config.seeds,
        "run_seed": config.run_seed,
        "n_drafts": config.n_drafts,
        "draft_only": config.draft_only,
        "max_tokens": config.max_tokens,
        "weights_revision": args.revision,
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "python_version": sys.version.split()[0],
        "workers": args.workers,
    }
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise SystemExit(f"Existing checkpoint settings differ: {manifest_file}")
    elif output.exists():
        raise SystemExit(f"Existing output has no checkpoint manifest: {output}")
    else:
        write_json(manifest_file, manifest)

    paths = {seed: checkpoints / f"{seed}.json" for seed in config.seeds}
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("collection") != manifest or result.get("is_mock") is not False:
            raise SystemExit(f"Existing output does not match the live run: {output}")
        return result

    missing = [seed for seed, path in paths.items() if not path.exists()]
    total = len(paths)
    print(f"{output}: {total - len(missing)}/{total} tasks completed", flush=True)

    def collect_task(seed):
        episode = run_cell(replace(config, seeds=[seed]))
        write_json(paths[seed], episode)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_task, seed) for seed in missing]
        for count, future in enumerate(as_completed(futures), total - len(missing) + 1):
            future.result()
            if count == 1 or count % 10 == 0 or count == total:
                print(f"{output}: {count}/{total} tasks completed", flush=True)

    episodes = [json.loads(path.read_text()) for path in paths.values()]
    result = {
        **episodes[0],
        "total_tokens": sum(ep["total_tokens"] for ep in episodes),
        "tasks": [ep["tasks"][0] for ep in episodes],
        "collection": manifest,
    }
    write_json(output, result)
    return result


def pilot_success(episode):
    successes = 0
    for row in episode["tasks"]:
        draft = row["drafts"][0]["decision"]
        decision = Decision(draft["seller"], draft["quantity"], tuple(draft["shared_fields"]))
        successes += classify_failure(task_from_json(row["task"]), decision)["success"]
    return successes / len(episode["tasks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "study"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get("PYTHONHASHSEED", "").isdecimal():
        parser.error("set a fixed PYTHONHASHSEED before starting Python")
    if args.workers < 1:
        parser.error("--workers must be positive")

    root = Path("results") / args.model_name
    common = dict(
        model_name=args.model_name, backend="live", endpoint=args.endpoint, model_id=args.model_id,
    )
    pilot = RunConfig(
        **common, market="benign", seeds=list(range(90001, 90201)), n_drafts=1, draft_only=True,
    )
    episode = collect_cell(pilot, root / "pilot" / "benign.json", args)
    success = pilot_success(episode)
    print(f"{args.model_name} pilot single-draft success: {success:.1%}", flush=True)
    if not 0.20 <= success <= 0.70:
        raise SystemExit("Pilot success outside the paper's 20–70% range")
    if args.stage == "pilot":
        return

    for market in MARKETS:
        for split, seeds in (("cal", range(1, 101)), ("test", range(10001, 10201))):
            config = RunConfig(**common, market=market, seeds=list(seeds))
            collect_cell(config, root / f"{market}_{split}.json", args)


if __name__ == "__main__":
    main()
