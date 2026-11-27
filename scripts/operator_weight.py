"""Replay effort when the operator gives the shopper's modeled loss less weight."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from scripts.calibration_sensitivity import TOKEN_PRICE
from scripts.collect_live import write_json
from scripts.distributed_effort import MODEL
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import action_means, best_for_budget, decision_point, interval, price_action_means, read_responses


PENALTY = .5
WEIGHTS = (1, .1, .01, .001, 0)


def compare(calibration, tests, menu=MENU):
    groups = defaultdict(list)
    for point in calibration:
        groups[point.state].append(point)
    pooled = price_action_means(action_means(calibration, menu, True), PENALTY, TOKEN_PRICE, 0)
    states = {state: price_action_means(action_means(rows, menu, True), PENALTY, TOKEN_PRICE, 0)
              for state, rows in groups.items()}
    reference = None
    result = []
    for weight in WEIGHTS:
        expected = {state: {action: (weight * loss, cost) for action, (loss, cost) in values.items()}
                    for state, values in {None: pooled, **states}.items()}
        losses, inference, failures = [], [], []
        by_seed = defaultdict(list)
        counts = Counter()
        for point in tests:
            action = best_for_budget(point.budget, expected.get(point.state, expected[None]), menu)
            overpay, risk, tokens, _, _, failed = point.actions[action]
            loss = overpay + PENALTY * risk
            losses.append(loss)
            inference.append(tokens * TOKEN_PRICE)
            failures.append(failed)
            by_seed[point.seed].append(loss)
            counts[action] += 1
        seed_losses = [fmean(by_seed[seed]) for seed in sorted(by_seed)]
        if reference is None:
            reference = seed_losses
        result.append({"weight": weight, "shopper_loss": fmean(losses), "operator_inference_cost": fmean(inference),
                       "failure_rate": fmean(failures), "action_counts": counts,
                       "shopper_loss_change": interval(seed_losses, reference)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    if plan["model"] != MODEL:
        parser.error("this replay uses the Qwen3-8B matched policy panel")
    reviews = read_responses(args.followup, "disclosure", plan, entries)
    if reviews is None:
        raise ValueError("Complete regime-matched reviews are required")
    results = {}
    for regime, responses in (("opaque", None), ("disclosed", reviews)):
        points = defaultdict(list)
        for entry in entries:
            points[entry[1]["split"]].append(decision_point(entry, regime, responses))
        results[regime] = compare(points["cal"], points["test"])
    write_json(args.output, {"model": MODEL, "penalty": PENALTY, "dollars_per_million": TOKEN_PRICE * 1_000_000,
                             "calibration_tasks": len(points["cal"]), "test_tasks": len(points["test"]), "regimes": results})


if __name__ == "__main__":
    main()
