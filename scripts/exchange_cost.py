"""Compare both seller-exchange arms against the same opening compliant price."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import classify_failure, evaluate, optimum, task_from_json
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort_report import task_entry
from scripts.effort_followup_report import interval, ranked
from scripts.seller_exchange import ARMS, load_rows


def purchase_or_referral(task, decision, anchor, penalty):
    outcome = classify_failure(task, decision)
    if outcome["violation"] not in ("none", "overpay"):
        return anchor * (1 + penalty)
    return evaluate(task, decision, "disclosed")["allin"]


def costs(row, regime, penalty):
    anchor = optimum(task_from_json(row["opening_task"]))["allin"]
    compared = []
    for arm, task_key in ARMS:
        record = row[arm]
        task, _, drafts, thought = task_entry(task_from_json(row[task_key]), record)
        decision = ranked(task, [*drafts[:2], thought[2048]], regime)
        spend = sum(tokens(call) for call in [*record["short"][:2], record["thinking"]["2048"]])
        if arm == "adaptive":
            spend += tokens(row["buyer_request"]) + tokens(row["seller_reply"])
        compared.append(purchase_or_referral(task, decision, anchor, penalty) + spend * 10 / 1_000_000)
    return tuple(compared)


def summarize(tests):
    result = {"test_tasks": len(tests), "costs": {}}
    for regime in ("opaque", "disclosed"):
        result["costs"][regime] = {}
        for penalty in (.2, .5):
            static, adaptive = zip(*(costs(row, regime, penalty) for row in tests))
            result["costs"][regime][str(penalty)] = {"static": fmean(static), "adaptive": fmean(adaptive),
                                                     "paired_savings": interval(static, adaptive)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tests = [row for row in load_rows(args.exchange) if row["split"] == "test"]
    write_json(args.output, summarize(tests))


if __name__ == "__main__":
    main()
