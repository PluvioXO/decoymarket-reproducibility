"""Resample calibration and test seeds for the incremental value of agreement."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path
from random import Random
from statistics import fmean

from scripts.calibration_sensitivity import DRAWS, clusters
from scripts.collect_live import write_json
from scripts.common_policy_replay import components
from scripts.distributed_effort_report import MENU
from scripts.effort_baselines import check_state
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import best_for_budget, decision_point, source_for


SCENARIOS = {
    "qwen3-4b-q4": ((.5, None),),
    "qwen3-8b-q4": ((.5, None),),
    "qwen3-14b-q4": ((.2, None), (0, 5), (0, 20)),
}


def replay(calibration, tests, penalty, overhead):
    menu = MENU if overhead is None else (*MENU, "refer")

    def values(point):
        return tuple(components(point, action, penalty, 0, overhead, 0)[:2] for action in menu)

    cal = [[(point.state, check_state(point), tuple((loss / point.budget, price) for loss, price in values(point)))
            for point in group] for group in clusters(calibration)]
    test = [[(point.state, check_state(point), point.budget, dict(zip(menu, (sum(row) for row in values(point)))))
             for point in group] for group in clusters(tests)]

    def means(rows):
        estimates = {}
        for action, parts in zip(menu, zip(*rows), strict=True):
            loss, price = zip(*parts)
            estimates[action] = fmean(loss), fmean(price)
        return estimates

    def deltas(selection):
        pooled, groups = [], defaultdict(list)
        for state, passed, row in chain.from_iterable(selection):
            pooled.append(row)
            groups[state].append(row)
            groups[passed].append(row)
        fallback = means(pooled)
        costs = {state: means(rows) for state, rows in groups.items()}
        choices, seed_savings, differences = {}, [], 0
        for group in test:
            savings = []
            for state, passed, budget, observed in group:
                key = state, budget
                if key not in choices:
                    choices[key] = (best_for_budget(budget, costs.get(passed, fallback), menu),
                                    best_for_budget(budget, costs.get(state, fallback), menu))
                simple, full = choices[key]
                savings.append(observed[simple] - observed[full])
                differences += simple != full
            seed_savings.append(fmean(savings))
        return seed_savings, differences

    initial, changed = deltas(cal)
    rng = Random(710312)
    estimates = []
    for _ in range(DRAWS):
        sample, _ = deltas(rng.choices(cal, k=len(cal)))
        estimates.append(fmean(rng.choices(sample, k=len(sample))))
    estimates.sort()
    return {"penalty": penalty, "referral_cost": overhead, "calibration_seeds": len(cal), "test_seeds": len(test),
            "difference": fmean(initial), "different_actions": changed, "replicates": DRAWS,
            "interval": [estimates[int(DRAWS * q)] for q in (.025, .975)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summaries = {}
    for model, scenarios in SCENARIOS.items():
        _, entries = load_panel(source_for(args.results, model))
        split = defaultdict(list)
        for entry in entries:
            split[entry[1]["split"]].append(decision_point(entry, "opaque", None))
        summaries[model] = []
        for penalty, overhead in scenarios:
            row = replay(split["cal"], split["test"], penalty, overhead)
            summaries[model].append(row)
            print(model, row, flush=True)
            write_json(args.output, summaries)


if __name__ == "__main__":
    main()
