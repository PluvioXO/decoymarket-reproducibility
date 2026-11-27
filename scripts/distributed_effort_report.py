"""Pool completed blocks and score the matched inference-budget experiment."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from decoymarket.benchmark import Decision, MARKETS, classify_failure, evaluate, generate_task, optimum, select
from decoymarket.estimators import no_review_curve
from scripts.auto_effort import DraftPair, scored_outcome, tokens
from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS, SHORT_DRAFTS, THINK_CAPS, cases_for_block, load_checkpoint
from scripts.effort_extensions import calibrated_policies, evaluate_policies

TOKEN_PRICE = 10 / 1_000_000
MENU = ("stop", "draft", *(f"think_{cap}" for cap in THINK_CAPS), "review")
COUNTS = (1, 2, 4, 8)


def load_blocks(root, plan):
    rows, coverage = [], {}
    for worker, spec in plan["workers"].items():
        first = spec["first_block"]
        blocks = 0
        for index in range(first, first + spec["blocks"]):
            path = root / "results" / worker / f"block-{index:06d}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            group = data["tasks"]
            actual = [(row["market"], row["seed"], row["split"]) for row in group]
            if (data["worker"], data["block"]) != (worker, index) or actual != cases_for_block(index):
                raise ValueError(f"Block does not match the plan: {path}")
            rows.extend(group)
            blocks += 1
        exit_path = root / "workers" / worker / "exit_code"
        coverage[worker] = {
            "planned_blocks": spec["blocks"], "completed_blocks": blocks,
            "exit_code": int(exit_path.read_text()) if exit_path.exists() else None,
        }
    if not rows:
        raise ValueError("No complete study blocks")
    return rows, coverage


def task_entry(task, row):
    records = row["short"]
    if len(records) != SHORT_DRAFTS or set(row["thinking"]) != {str(cap) for cap in THINK_CAPS}:
        raise ValueError(f"Incomplete action log for {task.task_id}")
    drafts = [Decision.from_json(record["decision"]) for record in records]
    thought = {cap: Decision.from_json(row["thinking"][str(cap)]["decision"]) for cap in THINK_CAPS}
    return task, row, drafts, thought


def task_entries(rows):
    for row in rows:
        yield task_entry(generate_task(row["seed"], row["market"]), row)


def selected_actions(entry):
    task, row, drafts, thought = entry
    first = drafts[:2]
    chosen = first[select(task, first, "opaque")]
    actions = {"stop": (chosen, 0)}
    additions = {"draft": (drafts[2], row["short"][2])}
    additions.update({f"think_{cap}": (thought[cap], row["thinking"][str(cap)]) for cap in THINK_CAPS})
    for action, (extra, record) in additions.items():
        options = [*first, extra]
        actions[action] = (options[select(task, options, "opaque")], tokens(record))
    review = row["review"]
    revised = Decision.from_json(review["decision"]) if review["parse_ok"] else chosen
    actions["review"] = (revised, tokens(review))
    return actions


def replay(entry, penalty):
    task, row, drafts, _ = entry
    price = optimum(task)["allin"]
    actions = selected_actions(entry)
    if evaluate(task, actions["stop"][0], "opaque")["passes"]:
        state = "agreement" if drafts[0] == drafts[1] else "disagreement"
    else:
        state = "failed_check"
    outcomes = {
        action: scored_outcome(task, decision, price, penalty, extra_tokens)
        for action, (decision, extra_tokens) in actions.items()
    }
    return DraftPair(state, task.mandate.budget, sum(tokens(item) for item in row["short"][:2]), outcomes)


def draft_frontier(entries):
    results = {}
    for regime in ("opaque", "disclosed"):
        data = defaultdict(list)
        for task, row, drafts, thought in entries:
            curve = no_review_curve(task, drafts, [tokens(item) for item in row["short"]], regime, COUNTS)
            for count, value in curve.items():
                data[f"short_{count}"].append((value["success"], value["cost"]))
            for cap, decision in thought.items():
                score = classify_failure(task, decision)
                data[f"think_{cap}"].append((float(score["success"]), tokens(row["thinking"][str(cap)])))
        frontier = {
            name: {"optimal_rate": mean(success for success, _ in values), "mean_tokens": mean(cost for _, cost in values)}
            for name, values in data.items()
        }
        results[regime] = frontier
    finished = {
        str(cap): {
            "truncation_rate": mean(row["thinking"][str(cap)]["finish_reason"] == "length" for _, row, _, _ in entries),
            "parse_rate": mean(row["thinking"][str(cap)]["parse_ok"] for _, row, _, _ in entries),
        }
        for cap in THINK_CAPS
    }
    return {"frontiers": results, "thinking": finished}


def break_even_budgets(estimates):
    budgets = {}
    for state, actions in estimates.items():
        stop_loss = actions["stop"]["loss_per_budget_dollar"]
        budgets[state] = {}
        for name, action in actions.items():
            if name == "stop":
                continue
            saved = stop_loss - action["loss_per_budget_dollar"]
            budgets[state][name] = TOKEN_PRICE * action["extra_tokens"] / saved if saved > 0 else None
    return budgets


def compare(calibration, tests):
    scenarios = {}
    for penalty in (0.20, 0.10, 0.50):
        cal = [[replay(entry, penalty)] for entry in calibration]
        test = [[replay(entry, penalty)] for entry in tests]
        policies, fit = calibrated_policies(cal, MENU, TOKEN_PRICE)
        policies["agree2"] = lambda pair: "stop" if pair.state == "agreement" else "review"
        contrasts = ()
        if penalty == 0.20:
            contrasts = (("auto", "best_fixed"), ("auto", "budget_only"), ("auto", "agree2"),
                         ("think_1024", "draft"), ("think_2048", "draft"), ("review", "draft"))
        combined = evaluate_policies(test, policies, MENU, TOKEN_PRICE, contrasts)
        by_market = {}
        for market in MARKETS:
            subset = [pair for entry, pair in zip(tests, test) if entry[1]["market"] == market]
            by_market[market] = evaluate_policies(subset, policies, MENU, TOKEN_PRICE, ())
        scenarios[str(penalty)] = {
            **fit, "break_even_budget_vs_stop": break_even_budgets(fit["calibration_estimates"]),
            "overall": combined, "by_market": by_market,
        }
    return scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    checkpoint = load_checkpoint(root)
    if any(plan[key] != checkpoint[key] for key in CHECKPOINT_KEYS):
        raise ValueError("Unexpected model in study plan")
    if args.wait:
        while time.time() < plan["deadline_unix"] + 60:
            if all((root / "workers" / name / "exit_code").exists() for name in plan["workers"]):
                break
            time.sleep(30)
    rows, coverage = load_blocks(root, plan)
    splits = defaultdict(list)
    for entry in task_entries(rows):
        splits[entry[1]["split"]].append(entry)
    tests = splits["test"]
    summary = {
        "plan": plan, "worker_coverage": coverage,
        "calibration_tasks": len(splits["cal"]), "test_tasks": len(tests),
        "descriptive": {
            market: draft_frontier([entry for entry in tests if entry[1]["market"] == market])
            for market in MARKETS
        },
        "dollars_per_million_tokens": 10,
        "scenarios_by_violation_penalty": compare(splits["cal"], tests),
        "generated_unix": time.time(),
    }
    write_json(root / "summary.json", summary)
    print(f"Scored {len(tests)} held-out tasks. Results in {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
