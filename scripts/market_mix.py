"""Reweight the matched evaluation and calibration to different workloads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import fmean

from decoymarket.benchmark import MARKETS
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import action_means, best_for_budget, decision_point, price_action_means, read_responses


PENALTY = .5
TOKEN_PRICE = 10 / 1_000_000


def weights(note_share):
    return {"benign": (1 - note_share) / 2, "decoy": (1 - note_share) / 2, "decoy_injection": note_share}


def estimates(points, mix):
    if mix is None:
        return price_action_means(action_means(points, MENU, True), PENALTY, TOKEN_PRICE, 0)
    weighted = [(point, mix[point.market]) for point in points if mix[point.market] > 0]
    if not weighted:
        return None
    losses = {}
    for action in MENU:
        entries = []
        for point, weight in weighted:
            pay, risk, tokens, _, _, _ = point.actions[action]
            entries.append((pay / point.budget, risk / point.budget, tokens, weight))
        payments, risks, token_counts, observation_weights = zip(*entries)
        loss = fmean(payments, weights=observation_weights) + PENALTY * fmean(risks, weights=observation_weights)
        inference = TOKEN_PRICE * fmean(token_counts, weights=observation_weights)
        losses[action] = (loss, inference)
    return losses


def fit(calibration, mix):
    by_state = defaultdict(list)
    for point in calibration:
        by_state[point.state].append(point)
    pooled = estimates(calibration, mix)
    conditional = {state: estimates(points, mix) or pooled for state, points in by_state.items()}
    return pooled, conditional


def evaluate(tests, fitted):
    pooled, conditional = fitted
    savings = defaultdict(dict)
    for point in tests:
        budget_only = best_for_budget(point.budget, pooled, MENU)
        auto = best_for_budget(point.budget, conditional.get(point.state, pooled), MENU)
        pay, risk, tokens, _, _, _ = point.actions[budget_only]
        other_pay, other_risk, other_tokens, _, _, _ = point.actions[auto]
        savings[point.seed][point.market] = (pay + PENALTY * risk + TOKEN_PRICE * tokens
                                              - (other_pay + PENALTY * other_risk + TOKEN_PRICE * other_tokens))
    if any(set(row) != set(MARKETS) for row in savings.values()):
        raise ValueError("Each test seed must contain all three markets")
    return list(savings.values())


def average(rows, mix):
    return fmean(sum(row[market] * mix[market] for market in MARKETS) for row in rows)


def crossing(non_note, note):
    if non_note > 0:
        return 0.0
    return -non_note / (note - non_note) if note > 0 else None


def summarize(calibration, tests):
    base = evaluate(tests, fit(calibration, None))
    rows = []
    for share in (0, .05, .10, .15, .25, 1/3, .50, .75, 1):
        mix = weights(share)
        retuned = base if share == 1/3 else evaluate(tests, fit(calibration, mix))
        rows.append({"note_share": share, "fixed_calibration_savings": average(base, mix),
                     "retuned_calibration_savings": average(retuned, mix)})
    rng = Random(380243)
    threshold = crossing(average(base, weights(0)), average(base, weights(1)))
    draws = []
    for _ in range(2000):
        sampled = rng.choices(base, k=len(base))
        share = crossing(average(sampled, weights(0)), average(sampled, weights(1)))
        if share is not None:
            draws.append(share)
    draws.sort()
    bounds = [draws[50], draws[1950]] if len(draws) == 2000 else None
    return {"scenarios": rows, "fixed_calibration_break_even": threshold, "finite_crossing_draws": len(draws),
            "fixed_calibration_break_even_interval": bounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    if plan["model"] != "qwen3-8b-q4":
        raise ValueError("The workload comparison uses the matched 8B panel")
    counts = Counter((row["split"], task.market) for task, row, _, _ in entries)
    if any(counts["cal", m] != 208 or counts["test", m] != 832 for m in MARKETS):
        raise ValueError("The workload comparison requires the complete balanced source panel")
    reviews = read_responses(args.followup, "disclosure", plan, entries)
    if reviews is None:
        raise ValueError("The workload comparison requires completed disclosed reviews")
    report = {}
    for regime in ("opaque", "disclosed"):
        points = {split: [decision_point(entry, regime, reviews if regime == "disclosed" else None)
                          for entry in entries if entry[1]["split"] == split] for split in ("cal", "test")}
        report[regime] = summarize(points["cal"], points["test"])
    write_json(args.output, report)


if __name__ == "__main__":
    main()
