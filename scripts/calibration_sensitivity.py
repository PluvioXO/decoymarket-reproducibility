"""Refit the effort policies when resampling calibration and held-out tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path
from random import Random
from statistics import fmean

from decoymarket.benchmark import MARKETS
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import (action_means, best_for_budget, decision_point,
                                            price_action_means, read_responses)


PENALTIES = {"qwen3-8b-q4": .5, "qwen3-14b-q4": .2}
TOKEN_PRICE = 10 / 1_000_000
DRAWS = 2000


def clusters(points):
    groups = defaultdict(list)
    for point in points:
        groups[point.seed].append(point)
    for members in groups.values():
        if len(members) != len(MARKETS) or {point.market for point in members} != set(MARKETS):
            raise ValueError("Calibration resampling requires one observation for each seed and market")
    return [groups[seed] for seed in sorted(groups)]


def fit_and_compare(calibration, trials, penalty, menu=MENU):
    states = defaultdict(list)
    for point in calibration:
        states[point.state].append(point)
    pooled = price_action_means(action_means(calibration, menu, True), penalty, TOKEN_PRICE, 0)
    conditional = {state: price_action_means(action_means(rows, menu, True), penalty, TOKEN_PRICE, 0)
                   for state, rows in states.items()}
    changes = []
    for point, costs in trials:
        budget_only = best_for_budget(point.budget, pooled, menu)
        auto = best_for_budget(point.budget, conditional.get(point.state, pooled), menu)
        changes.append((costs[budget_only] - costs[auto], auto == budget_only))
    return fmean(change for change, _ in changes), all(same for _, same in changes)


def nested_replay(calibration, tests, penalty, *, menu=MENU):
    observed = []
    for point in tests:
        selected = {}
        for action, (pay, risk, tokens, _, _, _) in point.actions.items():
            selected[action] = pay + penalty * risk + TOKEN_PRICE * tokens
        observed.append(selected)
    by_seed = defaultdict(list)
    for point, costs in zip(tests, observed, strict=True):
        by_seed[point.seed].append((point, costs))
    cal_groups = clusters(calibration)
    test_groups = [by_seed[group[0].seed] for group in clusters(tests)]
    point, same = fit_and_compare(calibration, list(zip(tests, observed, strict=True)), penalty, menu)
    rng = Random(412527)
    estimates, agreements = [], []
    for _ in range(DRAWS):
        cal_sample = list(chain.from_iterable(rng.choices(cal_groups, k=len(cal_groups))))
        test_sample = chain.from_iterable(rng.choices(test_groups, k=len(test_groups)))
        estimate, all_same = fit_and_compare(cal_sample, test_sample, penalty, menu)
        estimates.append(estimate)
        agreements.append(all_same)
    estimates.sort()
    return {"penalty": penalty, "calibration_seeds": len(cal_groups), "test_seeds": len(test_groups),
            "bootstrap_draws": DRAWS, "point": point, "same_actions": same,
            "interval": [estimates[int(DRAWS * .025)], estimates[int(DRAWS * .975)]],
            "draws_with_same_actions": sum(agreements)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("regime", choices=("opaque", "disclosed"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    penalty = PENALTIES[plan["model"]]
    reviews = None
    if args.regime == "disclosed":
        reviews = read_responses(args.followup, "disclosure", plan, entries)
        if reviews is None:
            raise ValueError("Complete regime-matched reviews are required")
    points = defaultdict(list)
    for entry in entries:
        points[entry[1]["split"]].append(decision_point(entry, args.regime, reviews))
    report = nested_replay(points["cal"], points["test"], penalty)
    write_json(args.output, {"model": plan["model"], "regime": args.regime, **report})


if __name__ == "__main__":
    main()
