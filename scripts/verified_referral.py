"""Replay effort policies under a verified execution gate with a fixed referral cost."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from scripts.calibration_sensitivity import TOKEN_PRICE
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import best_for_budget, decision_point, interval, read_responses


ACTIONS = (*MENU, "refer")


def with_referral(point):
    return replace(point, actions={**point.actions, "refer": (0, 0, 0, 0, False, True)})


def costs_for(points, overhead, *, scale_by_budget):
    costs = {}
    for action in ACTIONS:
        values = []
        for point in points:
            overpay, _, tokens, _, _, referred = point.actions[action]
            scale = point.budget if scale_by_budget else 1
            values.append((overpay / scale, overhead * referred + TOKEN_PRICE * tokens))
        overpay, referral_and_inference = zip(*values)
        costs[action] = (fmean(overpay), fmean(referral_and_inference))
    return costs


def replay(calibration, tests, overhead):
    grouped = defaultdict(list)
    for point in calibration:
        grouped[point.state].append(point)
    pooled = costs_for(calibration, overhead, scale_by_budget=True)
    state_costs = {state: costs_for(points, overhead, scale_by_budget=True) for state, points in grouped.items()}
    fixed_costs = costs_for(calibration, overhead, scale_by_budget=False)
    fixed = min(ACTIONS, key=lambda action: sum(fixed_costs[action]))
    by_seed = defaultdict(lambda: defaultdict(list))
    counts, referrals = defaultdict(Counter), Counter()
    for point in tests:
        selected = {
            "auto": best_for_budget(point.budget, state_costs.get(point.state, pooled), ACTIONS),
            "budget_only": best_for_budget(point.budget, pooled, ACTIONS),
            "best_fixed": fixed,
            "always_refer": "refer",
        }
        for name, action in selected.items():
            overpay, _, tokens, _, _, referred = point.actions[action]
            by_seed[name][point.seed].append(overpay + overhead * referred + TOKEN_PRICE * tokens)
            counts[name][action] += 1
            referrals[name] += referred
    seeds = sorted(by_seed["auto"])
    costs = {name: [fmean(per_seed[seed]) for seed in seeds] for name, per_seed in by_seed.items()}
    return {"cost": {name: fmean(values) for name, values in costs.items()},
            "referral_rate": {name: referrals[name] / len(tests) for name in costs},
            "actions": dict(counts),
            "auto_savings": {name: interval(values, costs["auto"]) for name, values in costs.items() if name != "auto"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    counts = Counter(row["split"] for _, row, _, _ in entries)
    if plan["model"] != "qwen3-8b-q4" or counts != {"cal": 624, "test": 2496}:
        raise ValueError("Verified referral requires the complete matched Qwen3-8B panel")
    reviews = read_responses(args.followup, "disclosure", plan, entries)
    if reviews is None:
        raise ValueError("Verified referral requires the regime-matched reviews")
    summary = {"calibration_tasks": counts["cal"], "test_tasks": counts["test"], "checks": {}}
    for regime in ("opaque", "disclosed"):
        points = defaultdict(list)
        saved = reviews if regime == "disclosed" else None
        for entry in entries:
            points[entry[1]["split"]].append(with_referral(decision_point(entry, regime, saved)))
        summary["checks"][regime] = {str(cost): replay(points["cal"], points["test"], cost) for cost in (1, 5, 20)}
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
