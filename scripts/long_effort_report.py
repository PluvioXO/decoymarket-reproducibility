"""Replay adaptive effort when a 4,096-token thinking action is available."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU
from scripts.effort_followup import LONG_MENU_BLOCKS, load_panel
from scripts.effort_followup_report import decision_point, interval, read_responses, replay


def extended_entry(entry, records):
    task, row, drafts, thought = entry
    call_records = {**row["thinking"], **records}
    choices = {**thought, **{int(cap): Decision.from_json(call["decision"]) for cap, call in records.items()}}
    return task, {**row, "thinking": call_records}, drafts, choices


def summarize(entries, calls, reviews):
    subset = [extended_entry(entry, calls[entry[0].task_id]) for entry in entries if entry[0].task_id in calls]
    counts = {split: sum(row["split"] == split for _, row, _, _ in subset) for split in ("cal", "test")}
    if counts != {"cal": LONG_MENU_BLOCKS * 6, "test": LONG_MENU_BLOCKS * 24}:
        raise ValueError(f"Incomplete paired long-menu panel: {counts}")
    summary = {"calibration_tasks": counts["cal"], "test_tasks": counts["test"], "regimes": {}}
    thinking = {cap: [row["thinking"][cap] for _, row, _, _ in subset if row["split"] == "test"]
                for cap in ("2048", "4096")}
    summary["cutoff_rates"] = {cap: fmean(call["finish_reason"] == "length" for call in rows)
                               for cap, rows in thinking.items()}
    summary["mean_calls"] = {cap: {"tokens": fmean(tokens(call) for call in rows),
                                   "seconds": fmean(call["latency_s"] for call in rows)}
                             for cap, rows in thinking.items()}
    scenarios = [(penalty, 10, 0) for penalty in (.1, .2, .5)]
    scenarios += [(penalty, 10, delay) for penalty in (.2, .5) for delay in (.01, .05)]
    for regime in ("opaque", "disclosed"):
        saved_reviews = reviews if regime == "disclosed" else None
        points = {split: [decision_point(entry, regime, saved_reviews) for entry in subset if entry[1]["split"] == split]
                  for split in counts}
        standard = replay(points["cal"], points["test"], MENU, scenarios, keep_seed_costs=True)
        extended = replay(points["cal"], points["test"], (*MENU, "think_4096"), scenarios, keep_seed_costs=True)
        savings = []
        for before, after in zip(standard, extended, strict=True):
            standard_costs, extended_costs = before.pop("seed_costs"), after.pop("seed_costs")
            savings.append({**{key: before[key] for key in ("penalty", "dollars_per_million", "dollars_per_second")},
                            **{policy: interval(costs, extended_costs[policy])
                               for policy, costs in standard_costs.items()}})
        summary["regimes"][regime] = {"standard": standard, "extended": extended, "menu_savings": savings}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("long_menu", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, entries = load_panel(args.source)
    if plan["model"] != "qwen3-8b-q4":
        raise ValueError("The long-menu comparison uses the matched Qwen3-8B panel")
    calls = read_responses(args.long_menu, "long-menu", plan, entries)
    reviews = read_responses(args.followup, "disclosure", plan, entries)
    if calls is None or reviews is None:
        raise ValueError("Both paired completion caps and matched disclosed reviews are required")
    write_json(args.output, summarize(entries, calls, reviews))


if __name__ == "__main__":
    main()
