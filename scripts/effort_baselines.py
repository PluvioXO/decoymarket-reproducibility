"""Compare the full effort rule with check-only calibration and AGREE2."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from scripts.calibration_sensitivity import PENALTIES, TOKEN_PRICE
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import (action_means, best_for_budget, decision_point, interval,
                                            price_action_means, read_responses)


def check_state(point):
    return point.state != "failed_check"


def fit(calibration, condition, penalty, menu=MENU):
    groups = defaultdict(list)
    for point in calibration:
        groups[condition(point)].append(point)
    return {state: price_action_means(action_means(rows, menu, True), penalty, TOKEN_PRICE, 0)
            for state, rows in groups.items()}


def compare(calibration, trials, penalty, *, menu=MENU):
    pooled = price_action_means(action_means(calibration, menu, True), penalty, TOKEN_PRICE, 0)
    full = fit(calibration, lambda point: point.state, penalty, menu)
    checked = fit(calibration, check_state, penalty, menu)
    by_policy = defaultdict(lambda: defaultdict(list))
    actions = defaultdict(Counter)
    failed = defaultdict(int)
    for point in trials:
        choices = {
            "auto": best_for_budget(point.budget, full.get(point.state, pooled), menu),
            "check_only": best_for_budget(point.budget, checked.get(check_state(point), pooled), menu),
            "budget_only": best_for_budget(point.budget, pooled, menu),
            "agree2": "stop" if point.state == "agreement" else "review",
        }
        for name, action in choices.items():
            payment, risk, tokens, _, _, failure = point.actions[action]
            by_policy[name][point.seed].append(payment + penalty * risk + TOKEN_PRICE * tokens)
            actions[name][action] += 1
            failed[name] += failure
    seeds = sorted(by_policy["auto"])
    costs = {name: [fmean(values[seed]) for seed in seeds] for name, values in by_policy.items()}
    return {"cost": {name: fmean(values) for name, values in costs.items()},
            "actions": {name: dict(counts) for name, counts in actions.items()},
            "failure_rates": {name: count / len(trials) for name, count in failed.items()},
            "auto_savings": {name: interval(costs[name], costs["auto"]) for name in costs if name != "auto"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    penalty = PENALTIES[plan["model"]]
    reviews = read_responses(args.followup, "disclosure", plan, entries)
    if reviews is None:
        raise ValueError("The heuristic comparison requires completed disclosed reviews")
    outcome = {"model": plan["model"], "penalty": penalty, "report": {}}
    for regime in ("opaque", "disclosed"):
        points = defaultdict(list)
        for entry in entries:
            point = decision_point(entry, regime, reviews if regime == "disclosed" else None)
            points[entry[1]["split"]].append(point)
        outcome["report"][regime] = compare(points["cal"], points["test"], penalty)
    write_json(args.output, outcome)


if __name__ == "__main__":
    main()
