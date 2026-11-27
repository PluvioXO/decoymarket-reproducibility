"""Compare effort rules on all archived full-menu panels and under a shared execution gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import fmean

from scripts.calibration_sensitivity import TOKEN_PRICE, clusters
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU, load_blocks, task_entries
from scripts.effort_baselines import check_state
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import best_for_budget, decision_point, interval, read_responses, source_for


PENALTIES = (.1, .2, .5)
DELAYS = (0, .01, .05)
REFERRALS = (1, 5, 20)
LOOKUPS = (0, .5)


def components(point, action, penalty, delay, overhead, lookup):
    if action == "refer":
        return 0, overhead, False, True, 0, 0
    overpay, exposure, tokens, seconds, _, failed = point.actions[action]
    loss = overpay + penalty * exposure
    expense = TOKEN_PRICE * tokens + delay * seconds
    if overhead is None:
        return loss, expense, failed, False, tokens, seconds
    return overpay, expense + lookup + overhead * failed, False, failed, tokens, seconds


def replay(calibration, tests, penalty, delay, *, overhead=None, lookup=0):
    menu = MENU if overhead is None else (*MENU, "refer")
    groups = defaultdict(list)
    for point in calibration:
        groups[check_state(point)].append(point)
        groups[point.state].append(point)

    def fit(points, normalize):
        estimates = {}
        for action in menu:
            values = []
            for point in points:
                loss, expense, *_ = components(point, action, penalty, delay, overhead, lookup)
                values.append((loss / (point.budget if normalize else 1), expense))
            estimates[action] = tuple(map(fmean, zip(*values)))
        return estimates

    pooled = fit(calibration, True)
    fixed = best_for_budget(1, fit(calibration, False), menu)
    conditional = {state: fit(points, True) for state, points in groups.items()}
    seed_costs, observations = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    action_counts = defaultdict(Counter)
    action_differences = Counter()
    for point in tests:
        choices = {
            "fixed": fixed,
            "budget_only": best_for_budget(point.budget, pooled, menu),
            "budget_check": best_for_budget(point.budget, conditional.get(check_state(point), pooled), menu),
            "auto": best_for_budget(point.budget, conditional.get(point.state, pooled), menu),
        }
        if overhead is not None:
            choices["direct_referral"] = "refer"
        for policy, action in choices.items():
            loss, expense, failed, referred, tokens, seconds = components(point, action, penalty, delay, overhead, lookup)
            action_counts[policy][action] += 1
            action_differences[policy] += action != choices["auto"]
            seed_costs[policy][point.seed].append(loss + expense)
            observations[policy].append((loss + expense, loss, expense, failed, referred, tokens, seconds))

    metrics = ("cost", "shopper_loss", "operation_cost", "failure_rate", "referral_rate", "extra_tokens", "extra_seconds")
    output = {}
    for policy, rows in observations.items():
        output[policy] = dict(zip(metrics, map(fmean, zip(*rows))))
        output[policy]["actions"] = dict(action_counts[policy])
        output[policy]["different_from_auto"] = action_differences[policy]
        if overhead is not None:
            referral_rate = output[policy]["referral_rate"]
            output[policy]["completion_at_referral_success"] = {
                str(success): 1 - referral_rate * (1 - success) for success in (.8, .9, 1)
            }

    if delay == 0 and lookup == 0:
        seeds = sorted(seed_costs["auto"])
        means = {policy: [fmean(rows[seed]) for seed in seeds] for policy, rows in seed_costs.items()}
        savings = {policy: interval(values, means["auto"]) for policy, values in means.items() if policy != "auto"}
    else:
        savings = {policy: {"delta": row["cost"] - output["auto"]["cost"]}
                   for policy, row in output.items() if policy != "auto"}
    return {"penalty": penalty, "delay": delay, "referral_cost": overhead, "lookup_cost": lookup,
            "policies": output, "auto_savings": savings}


def panel(entries, regime, reviews):
    split = defaultdict(list)
    for entry in entries:
        split[entry[1]["split"]].append(decision_point(entry, regime, reviews))
    for points in split.values():
        clusters(points)
    cal, tests = split["cal"], split["test"]
    print(f"{regime}: {len(cal)} calibration, {len(tests)} held out", flush=True)
    return {
        "calibration": len(cal), "test": len(tests),
        "proposal_cost": [replay(cal, tests, penalty, delay) for penalty, delay in product(PENALTIES, DELAYS)],
        "execution_cost": [replay(cal, tests, 0, delay, overhead=overhead, lookup=lookup)
                           for delay, overhead, lookup in product(DELAYS, REFERRALS, LOOKUPS)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    followup = args.results / "effort_followup_2026-11-25"
    root = source_for(args.results, "qwen3-8b-q4")
    plan = json.loads((root / "plan.json").read_text())
    rows, _ = load_blocks(root, plan)
    if Counter(row["split"] for row in rows) != {"cal": 1686, "test": 6744}:
        raise ValueError("Full-panel replay requires all original Qwen3-8B blocks")
    output = {"full_8b": {"opaque": panel(task_entries(rows), "opaque", None)}, "matched": {}}
    write_json(args.output, output)
    for model in ("qwen3-4b-q4", "qwen3-8b-q4", "qwen3-14b-q4"):
        print(model, flush=True)
        plan, entries = load_panel(source_for(args.results, model))
        reviews = read_responses(followup, "disclosure", plan, entries)
        if reviews is None:
            raise ValueError(f"Missing matched reviews for {model}")
        output["matched"][model] = {regime: panel(entries, regime, reviews if regime == "disclosed" else None)
                                     for regime in ("opaque", "disclosed")}
        write_json(args.output, output)

    for model, penalty in (("qwen3-8b-q4", .5), ("qwen3-14b-q4", .2)):
        path = args.results / "robustness_followup_2026-11-25" / f"baselines-{model}.json"
        reference = json.loads(path.read_text())["report"]["opaque"]["cost"]
        row = next(row for row in output["matched"][model]["opaque"]["proposal_cost"]
                   if (row["penalty"], row["delay"]) == (penalty, 0))
        for old, new in (("check_only", "budget_check"), ("budget_only", "budget_only"), ("auto", "auto")):
            if abs(reference[old] - row["policies"][new]["cost"]) > 1e-9:
                raise ValueError(f"Replay failed to recover the previous {model} {old} result")
    print("Both prior baseline comparisons reproduced", flush=True)


if __name__ == "__main__":
    main()
