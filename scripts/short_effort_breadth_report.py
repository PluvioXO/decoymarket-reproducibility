"""Combine the matched 14-checkpoint short-action policy replays."""
from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import median

from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS
from scripts.model_breadth_report import MODELS
from scripts.short_effort_transfer import BLOCKS, GPT_OSS, REGIMES, SCENARIOS, caps


STUDY = "short_effort_breadth_2026-11-26"
ORIGINAL = ("qwen3-8b-q4", "gemma-3-4b-q4", "mistral-7b-v03-q4")


def highlights(models):
    output = {}
    for regime in REGIMES:
        output[regime] = {}
        for index, (penalty, _, _) in enumerate(SCENARIOS):
            rows = [model["regimes"][regime][index] for model in models]
            estimates = [row["refitted"] for row in rows]
            points = [item["point"] for item in estimates]
            output[regime][str(penalty)] = {
                "savings_range": [min(points), max(points)],
                "median_savings": median(points),
                "intervals_above_zero": sum(item["interval"][0] > 0 for item in estimates),
                "intervals_below_zero": sum(item["interval"][1] < 0 for item in estimates),
                "check_matches_auto": sum(isclose(row["heuristics"]["cost"]["auto"], row["heuristics"]["cost"]["check_only"],
                                                 abs_tol=1e-9, rel_tol=0) for row in rows),
            }
    return output


def merge(results):
    breadth = json.loads((results / "model_breadth_2026-11-26" / "summary.json").read_text())
    checkpoints = {row["model"]: row for row in breadth["models"]}
    models = []
    for model, (label, family) in MODELS.items():
        path = (results / "review_loop_2026-11-25" / f"short-effort-{model}.json" if model in ORIGINAL
                else results / STUDY / "calls" / model / "summary.json")
        row = json.loads(path.read_text())
        if any(row["checkpoint"][key] != checkpoints[model][key] for key in CHECKPOINT_KEYS):
            raise ValueError(f"Wrong checkpoint for {model} in {path}")
        if (row["calibration_tasks"], row["test_tasks"], row["blocks"]) != (108, 432, list(BLOCKS)):
            raise ValueError(f"Wrong short-action panel in {path}")
        if model == GPT_OSS and (row["draft_cap"], row["review_cap"], row["reasoning_effort"]) != (768, 768, "low"):
            raise ValueError(f"Wrong native reasoning profile in {path}")
        if model != GPT_OSS and row.get("review_cap", 256) != 256:
            raise ValueError(f"Wrong review cap in {path}")
        for regime in REGIMES:
            if [(s["penalty"], s["dollars_per_million"], s["dollars_per_second"])
                    for s in row["regimes"][regime]] != list(SCENARIOS):
                raise ValueError(f"Wrong policy prices in {path}")
            for scenario in row["regimes"][regime]:
                estimate = scenario["refitted"]
                if (estimate["calibration_seeds"], estimate["test_seeds"], estimate["bootstrap_draws"]) != (36, 144, 2000):
                    raise ValueError(f"Wrong paired interval design in {path}")
                if not isclose(estimate["point"], scenario["budget_only"]["cost"] - scenario["auto"]["cost"],
                               abs_tol=1e-9, rel_tol=0):
                    raise ValueError(f"Policy savings do not match the action replay in {path}")
        models.append({"model": model, "label": label, "family": family, "draft_cap": caps(model)[0], **row})
    return {"models": models, "highlights": highlights(models)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    write_json(args.results / STUDY / "summary.json", merge(args.results))


if __name__ == "__main__":
    main()
