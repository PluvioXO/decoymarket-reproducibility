"""Rerank saved short drafts with both paired thinking caps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision, MARKETS
from scripts.collect_live import write_json
from scripts.effort_followup import load_panel
from scripts.effort_followup_report import MODELS, interval, ranked, read_responses, score, source_for


def audit(entries, replies):
    panels = defaultdict(list)
    for task, _, drafts, _ in entries:
        if task.task_id not in replies:
            continue
        outcomes = {}
        for cap in ("2048", "4096"):
            thinking = Decision.from_json(replies[task.task_id][cap]["decision"])
            pool = [*drafts[:2], thinking]
            opaque = ranked(task, pool, "opaque")
            disclosed = ranked(task, pool, "disclosed")
            alone, mixed, restored = (score(task, pick) for pick in (thinking, opaque, disclosed))
            outcomes[cap] = {"alone": alone, "opaque": mixed, "disclosed": restored,
                             "planted_displacements": alone and opaque.seller in task.decoy_sellers}
        panels[task.market].append(outcomes)
    if set(panels) != set(MARKETS) or any(len(rows) != 32 for rows in panels.values()):
        raise ValueError("Cap replay requires all 32 saved test tasks per market")
    return {market: summarize(rows) for market, rows in panels.items()}


def summarize(rows):
    result = {"tasks": len(rows)}
    harms = {}
    for cap in ("2048", "4096"):
        samples = [row[cap] for row in rows]
        alone = [sample["alone"] for sample in samples]
        opaque = [sample["opaque"] for sample in samples]
        disclosed = [sample["disclosed"] for sample in samples]
        displaced = sum(sample["planted_displacements"] for sample in samples)
        harms[cap] = [a - o for a, o in zip(alone, opaque)]
        result[cap] = {"alone": fmean(alone), "opaque": fmean(opaque), "disclosed": fmean(disclosed),
                       "planted_displacements": displaced, "opaque_minus_alone": interval(opaque, alone),
                       "disclosed_minus_opaque": interval(disclosed, opaque)}
    result["drop_4096_minus_2048"] = interval(harms["4096"], harms["2048"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {}
    for model in MODELS:
        plan, entries = load_panel(source_for(args.results, model))
        records = read_responses(args.followup, "caps", plan, entries)
        if records is None:
            raise ValueError(f"Missing paired-cap responses for {model}")
        results[model] = audit(entries, records)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
