"""Replay effort rules after a market shift and when referrals may go unresolved."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import MARKETS, optimum
from scripts.collect_live import write_json
from scripts.common_policy_replay import MENU, PENALTIES, components
from scripts.effort_baselines import check_state
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import best_for_budget, decision_point, interval, source_for


MODELS = ("qwen3-4b-q4", "qwen3-8b-q4", "qwen3-14b-q4")
POLICIES = ("budget_only", "budget_check", "auto")


def action_values(point, optimum_price, menu, penalty, overhead, resolution):
    choices = {}
    for action in menu:
        loss, expense, failure, referred, *_ = components(point, action, penalty, 0, overhead, 0)
        uncompleted = (1 - resolution) if referred else float(failure)
        if referred:
            loss += penalty * optimum_price * uncompleted
        choices[action] = (loss, expense, float(referred), uncompleted)
    return choices


def fit(observations, menu):
    buckets = defaultdict(list)
    for point, costs in observations:
        for group in (None, check_state(point), point.state):
            buckets[group].append((point.budget, costs))
    return {group: {action: (fmean(values[action][0] / budget for budget, values in rows),
                             fmean(values[action][1] for _, values in rows)) for action in menu}
            for group, rows in buckets.items()}


def evaluate(tests, estimates, menu):
    rows = defaultdict(list)
    for point, costs in tests:
        for policy, state in zip(POLICIES, (None, check_state(point), point.state), strict=True):
            action = best_for_budget(point.budget, estimates.get(state, estimates[None]), menu)
            rows[policy].append((point.seed, action, costs[action]))
    if "refer" in menu:
        rows["direct_referral"] = [(point.seed, "refer", costs["refer"]) for point, costs in tests]
    return rows


def summarize(rows, confidence):
    metrics = ("shopper_loss", "operation_cost", "referral_rate", "uncompleted")
    result, seeds = {}, defaultdict(lambda: defaultdict(list))
    for policy, values in rows.items():
        averages = dict(zip(metrics, map(fmean, zip(*(cost for _, _, cost in values))), strict=True))
        result[policy] = {"cost": averages["shopper_loss"] + averages["operation_cost"],
                          "completion": 1 - averages.pop("uncompleted"), **averages,
                          "actions": dict(Counter(action for _, action, _ in values))}
        for seed, _, (loss, expense, _, _) in values:
            seeds[policy][seed].append(loss + expense)
    keys = sorted(seeds["auto"])
    auto = [fmean(seeds["auto"][seed]) for seed in keys]
    savings = {policy: interval([fmean(seeds[policy][seed]) for seed in keys], auto) if confidence else
               {"delta": result[policy]["cost"] - result["auto"]["cost"]} for policy in rows if policy != "auto"}
    return {"policies": result, "auto_savings": savings}


def scenario(panel, penalty, overhead=None, resolution=1):
    menu = MENU if overhead is None else (*MENU, "refer")
    split = defaultdict(list)
    for point, partition, optimum_price in panel:
        split[partition].append((point, action_values(point, optimum_price, menu, penalty, overhead, resolution)))
    cal, tests = split["cal"], split["test"]
    pooled = fit(cal, menu)
    confidence = overhead is None and penalty in (.2, .5) or overhead == 20
    output = {"penalty": penalty, "overhead": overhead, "resolution": resolution,
              "pooled": summarize(evaluate(tests, pooled, menu), confidence), "held_out": {}}
    for market in MARKETS:
        train = [(point, values) for point, values in cal if point.market != market]
        trial = [(point, values) for point, values in tests if point.market == market]
        pooled_rows = evaluate(trial, pooled, menu)
        excluded_rows = evaluate(trial, fit(train, menu), menu)
        shift = None
        if confidence:
            samples = [[loss + expense for _, _, (loss, expense, _, _) in rows["auto"]]
                       for rows in (excluded_rows, pooled_rows)]
            shift = interval(*samples)
        output["held_out"][market] = {"calibration": len(train), "test": len(trial), "auto_cost_shift": shift,
                                      "pooled": summarize(pooled_rows, confidence), "excluded": summarize(excluded_rows, confidence)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = {}
    for model in MODELS:
        plan, entries = load_panel(source_for(args.results, model))
        panel = [(decision_point(entry, "opaque", None), entry[1]["split"], optimum(entry[0])["allin"]) for entry in entries]
        counts = Counter(partition for _, partition, _ in panel)
        if counts != {"cal": 624, "test": 2496}:
            raise ValueError(f"Incorrect matched-sample sizes for {model}")
        output[model] = {"checkpoint_sha256": plan["model_sha256"], "calibration": counts["cal"], "test": counts["test"],
                         "proposal": [scenario(panel, penalty) for penalty in PENALTIES],
                         "execution": [scenario(panel, .2, overhead, resolution)
                                       for overhead, resolution in product((5, 20), (.7, .9, 1))]}
        print(f"{model}: completed nine scenarios on {counts['test']} held-out tasks", flush=True)
        write_json(args.output, output)

    published = json.loads((args.results / "catalog_verification_2026-11-25" / "common-policy.json").read_text())
    for model, comparison in output.items():
        reference = next(row for row in published["matched"][model]["opaque"]["proposal_cost"]
                         if row["penalty"] == .2 and row["delay"] == 0)
        for policy in POLICIES:
            current = next(row for row in comparison["proposal"] if row["penalty"] == .2)["pooled"]["policies"][policy]
            if abs(current["cost"] - reference["policies"][policy]["cost"]) > 1e-9:
                raise ValueError(f"Pooled proposal replay does not match the reference for {model} {policy}")
    print("Original pooled proposal costs reproduced", flush=True)


if __name__ == "__main__":
    main()
