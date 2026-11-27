"""Score the matched follow-up and replay its cost and delay scenarios."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from statistics import fmean

from decoymarket.analysis import paired_delta_ci
from decoymarket.benchmark import Decision, MARKETS, classify_failure, evaluate, optimum, select
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort_report import MENU, load_blocks, task_entries
from scripts.effort_followup import jobs, load_panel

MODELS = ("qwen3-4b-q4", "qwen3-8b-q4", "qwen3-14b-q4")
PENALTIES = (0, .1, .2, .35, .5, .75, 1)
TOKEN_PRICES = (1, 10, 50)
DELAY_PRICES = (0, .01, .05, .10)


def source_for(results, model):
    if model == "qwen3-8b-q4":
        return results / "distributed_effort"
    return results / "distributed_model_sweep_2026-11-24" / model


def read_responses(root, experiment, plan, entries):
    folder = root / experiment / plan["model"]
    if not folder.exists():
        return None
    cases = jobs(entries, experiment)
    replies = {}
    spec_paths = sorted(folder.glob("*/manifest.json"))
    for path in spec_paths:
        spec = json.loads(path.read_text())
        shard = spec["shard"]
        expected = dict(list(cases.items())[shard::len(spec_paths)])
        digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        if (spec["shards"] != len(spec_paths) or not 0 <= shard < len(spec_paths)
                or spec["experiment"] != experiment or spec["calls_sha256"] != digest
                or any(spec[key] != plan[key] for key in ("model", "model_revision", "model_sha256", "engine_revision"))):
            raise ValueError(f"Unexpected follow-up manifest {path}")
        files = {item.stem: item for item in path.parent.glob("*.json") if item != path}
        if set(files) != set(expected) or set(files) & set(replies):
            raise ValueError(f"Incomplete or duplicate follow-up responses in {path.parent}")
        for key, calls in expected.items():
            record = json.loads(files[key].read_text())
            if (set(record) != set(calls)
                    or any((value["seed"], value["max_tokens"], value["thinking"]) != tuple(calls[name][1:])
                           for name, value in record.items())):
                raise ValueError(f"Responses do not match {files[key]}")
            replies[key] = record
    if set(replies) != set(cases):
        raise ValueError(f"Incomplete {experiment} in {folder}")
    return replies


def score(task, decision):
    return classify_failure(task, decision)["success"]


def ranked(task, pool, regime):
    return pool[select(task, pool, regime)]


def interval(first, second):
    if len(first) != len(second) or not first:
        raise ValueError("A paired interval requires aligned nonempty samples")
    delta, lo, hi = paired_delta_ci(first, second)
    return {"delta": delta, "low": lo, "high": hi}


def seed_interval(first, second, seeds):
    if len(first) != len(second) or len(first) != len(seeds):
        raise ValueError("Seed-clustered intervals require aligned outcomes")
    groups = defaultdict(list)
    for seed, a, b in zip(seeds, first, second):
        groups[seed].append(a - b)
    if {len(group) for group in groups.values()} != {len(MARKETS)}:
        raise ValueError("Each cap-ablation seed must include every market")
    differences = [fmean(groups[seed]) for seed in sorted(groups)]
    return interval(differences, [0] * len(differences))


def field_rank(task, pool, hidden_clause):
    def key(decision):
        check = evaluate(task, decision, "disclosed")
        failures = sum(clause != hidden_clause for clause in check["failed"])
        return bool(failures), failures, check["visible"], decision.seller

    return min(pool, key=key)


def selector_audit(entries):
    summary = {}
    for market in MARKETS:
        rows = [(t, row, drafts, thought[2048]) for t, row, drafts, thought in entries
                if t.market == market and row["split"] == "test"]
        summaries = defaultdict(list)
        planted_losses = recovered = recovered_planted = 0
        for task, row, drafts, thinking in rows:
            pool = [*drafts[:2], thinking]
            opaque = ranked(task, pool, "opaque")
            disclosed = ranked(task, pool, "disclosed")
            alone, combined, restored = (score(task, pick) for pick in (thinking, opaque, disclosed))
            summaries["long_alone"].append(alone)
            summaries["long_added_opaque"].append(combined)
            summaries["long_added_disclosed"].append(restored)
            summaries["disclose_allin_only"].append(score(task, field_rank(task, pool, "subscription")))
            summaries["disclose_subscription_only"].append(score(task, field_rank(task, pool, "allin_budget")))
            summaries["eight_short_disclosed"].append(score(task, ranked(task, drafts, "disclosed")))
            summaries["cutoff"].append(row["thinking"]["2048"]["finish_reason"] == "length")
            summaries["parsed"].append(row["thinking"]["2048"]["parse_ok"])
            planted_loss = bool(alone and not combined and opaque.seller in task.decoy_sellers)
            planted_losses += planted_loss
            recovered += bool(alone and not combined and restored)
            recovered_planted += bool(planted_loss and restored)
        summary[market] = {
            "tasks": len(rows), **{name: fmean(values) for name, values in summaries.items()},
            "paired_added_vs_alone": interval(summaries["long_added_opaque"], summaries["long_alone"]),
            "paired_disclosed_vs_opaque": interval(summaries["long_added_disclosed"], summaries["long_added_opaque"]),
            "paired_thinking_vs_short": interval(summaries["long_alone"], summaries["eight_short_disclosed"]),
            "planted_replacements_of_optima": planted_losses, "disclosure_restored_optima": recovered,
            "disclosure_restored_planted_replacements": recovered_planted,
        }
    return summary


@dataclass(frozen=True)
class DecisionPoint:
    seed: int
    market: str
    budget: int
    state: str
    actions: dict


def decision_point(entry, regime, reviews, *, recheck_review=False):
    task, row, drafts, thought = entry
    first = drafts[:2]
    stop = ranked(task, first, regime)
    state = "agreement" if first[0] == first[1] else "disagreement"
    if not evaluate(task, stop, regime)["passes"]:
        state = "failed_check"
    choices = {
        "stop": (stop, None),
        "draft": (ranked(task, [*first, drafts[2]], regime), row["short"][2]),
        **{f"think_{cap}": (ranked(task, [*first, item], regime), row["thinking"][str(cap)])
           for cap, item in thought.items()},
    }
    if regime == "opaque" or reviews is not None:
        review = row["review"] if not reviews or task.task_id not in reviews else reviews[task.task_id]["review"]
        revised = Decision.from_json(review["decision"]) if review["parse_ok"] else stop
        if recheck_review and state != "failed_check" and not evaluate(task, revised, regime)["passes"]:
            revised = stop
        choices["review"] = (revised, review)
    price = optimum(task)["allin"]
    actions = {}
    for name, (choice, call) in choices.items():
        outcome = classify_failure(task, choice)
        failed = outcome["violation"] not in ("none", "overpay")
        overpay = 0 if failed else (outcome["overpay"] or 0)
        actions[name] = (overpay, price if failed else 0,
                         tokens(call) if call else 0, call["latency_s"] if call else 0,
                         outcome["success"], failed)
    return DecisionPoint(task.seed, task.market, task.mandate.budget, state, actions)


def action_means(points, menu, normalize):
    means = {}
    for action in menu:
        observations = []
        for point in points:
            pay, penalty, token_count, delay, _, _ = point.actions[action]
            budget = point.budget if normalize else 1
            observations.append((pay / budget, penalty / budget, token_count, delay))
        means[action] = tuple(fmean(values) for values in zip(*observations))
    return means


def price_action_means(means, penalty, token_price, delay_price):
    return {key: (pay + penalty * risk, token_price * tok + delay_price * sec)
            for key, (pay, risk, tok, sec) in means.items()}


def best_for_budget(budget, costs, menu):
    return min(menu, key=lambda action: budget * costs[action][0] + costs[action][1])


def replay(calibration, tests, menu, scenarios, *, keep_seed_costs=False):
    fixed = action_means(calibration, menu, False)
    pooled = action_means(calibration, menu, True)
    states = {state: action_means([p for p in calibration if p.state == state], menu, True)
              for state in {p.state for p in calibration}}
    outcomes = []
    for penalty, dollars_per_million, delay_price in scenarios:
        token_price = dollars_per_million / 1_000_000
        primary = dollars_per_million == 10 and delay_price == 0 and penalty in (.1, .2, .5)

        pooled_costs = price_action_means(pooled, penalty, token_price, delay_price)
        state_costs = {state: price_action_means(means, penalty, token_price, delay_price)
                       for state, means in states.items()}
        fixed_costs = price_action_means(fixed, penalty, token_price, delay_price)
        best_fixed = min(menu, key=lambda a: sum(fixed_costs[a]))
        results = defaultdict(list)
        base_seed_cost = defaultdict(list)
        market_cost = defaultdict(list)
        counts = defaultdict(Counter)

        for point in tests:
            policies = {
                "best_fixed": best_fixed,
                "budget_only": best_for_budget(point.budget, pooled_costs, menu),
                "auto": best_for_budget(point.budget, state_costs.get(point.state, pooled_costs), menu),
            }
            for policy, action in policies.items():
                pay, risk, tok, sec, optimal, failed = point.actions[action]
                cost = pay + penalty * risk + token_price * tok + delay_price * sec
                results[policy].append((cost, optimal, failed, tok, sec))
                counts[policy][action] += 1
                if primary or keep_seed_costs:
                    base_seed_cost[policy, point.seed].append(cost)
                if primary:
                    market_cost[policy, point.market].append(cost)
        row = {"penalty": penalty, "dollars_per_million": dollars_per_million, "dollars_per_second": delay_price}
        metrics = ("cost", "optimal_rate", "failure_rate", "extra_tokens", "extra_seconds")
        for policy, values in results.items():
            row[policy] = {name: fmean(v) for name, v in zip(metrics, zip(*values))}
            row[policy]["action_counts"] = dict(counts[policy])
        if primary or keep_seed_costs:
            seeds = sorted({point.seed for point in tests})
            seed_costs = {policy: [fmean(base_seed_cost[policy, seed]) for seed in seeds] for policy in results}
        if primary:
            row["paired_savings"] = {policy: interval(seed_costs[policy], seed_costs["auto"])
                                     for policy in ("budget_only", "best_fixed")}
            row["by_market"] = {
                market: {policy: fmean(market_cost[policy, market]) for policy in results}
                for market in MARKETS
            }
        if keep_seed_costs:
            row["seed_costs"] = seed_costs
        outcomes.append(row)
    return outcomes


def note_metrics(task, first, second, thinking):
    pool = [first, second, thinking]
    return {
        "planted_short_draft_rate": fmean(draft.seller in task.decoy_sellers for draft in pool[:2]),
        "planted_thinking_rate": thinking.seller in task.decoy_sellers,
        "long_alone": score(task, thinking),
        "long_added_opaque": score(task, ranked(task, pool, "opaque")),
        "long_added_disclosed": score(task, ranked(task, pool, "disclosed")),
    }


def note_audit(entries, replies):
    original, control = defaultdict(list), defaultdict(list)
    for task, _, drafts, thought in entries:
        if task.task_id not in replies:
            continue
        records = replies[task.task_id]
        without = [Decision.from_json(records[name]["decision"]) for name in ("first", "second", "thinking")]
        for name, value in note_metrics(task, drafts[0], drafts[1], thought[2048]).items():
            original[name].append(value)
        for name, value in note_metrics(task, *without).items():
            control[name].append(value)
    return {
        "tasks": len(replies),
        "with_note": {key: fmean(values) for key, values in original.items()},
        "without_note": {key: fmean(values) for key, values in control.items()},
        "paired_without_vs_with": {key: interval(control[key], values) for key, values in original.items()},
    }


def cap_audit(entries, replies):
    results = {"2048": defaultdict(list), "4096": defaultdict(list)}
    short = []
    seeds = []
    finished = unchanged = 0
    for task, _, drafts, _ in entries:
        if task.task_id in replies:
            seeds.append(task.seed)
            short.append(score(task, ranked(task, drafts, "disclosed")))
            smaller, larger = (replies[task.task_id][cap] for cap in ("2048", "4096"))
            if smaller["finish_reason"] != "length":
                finished += 1
                unchanged += smaller["decision"] == larger["decision"]
        for cap, record in replies.get(task.task_id, {}).items():
            decision = Decision.from_json(record["decision"])
            outcome = classify_failure(task, decision)
            results[cap]["optimal_rate"].append(outcome["success"])
            results[cap]["cutoff_rate"].append(record["finish_reason"] == "length")
            results[cap]["parse_rate"].append(record["parse_ok"])
            results[cap]["failure_rate"].append(outcome["violation"] not in ("none", "overpay"))
            results[cap]["tokens"].append(tokens(record))
            results[cap]["latency_s"].append(record["latency_s"])
    return {
        "tasks": len(replies),
        "caps": {cap: {key: fmean(values) for key, values in metrics.items()} for cap, metrics in results.items()},
        "paired_4096_vs_2048": {key: seed_interval(values, results["2048"][key], seeds)
                                for key, values in results["4096"].items()},
        "saved_eight_short_optimal_rate": fmean(short),
        "paired_4096_vs_saved_eight_short": seed_interval(results["4096"]["optimal_rate"], short, seeds),
        "shorter_cap_finished": finished,
        "same_decision_when_shorter_cap_finished": unchanged,
    }


def analyze_panel(plan, entries, followup):
    disclosure = read_responses(followup, "disclosure", plan, entries)
    panel = {key: plan[key] for key in ("model", "model_revision", "model_sha256", "engine_revision")}
    panel["calibration_tasks"] = sum(row["split"] == "cal" for _, row, _, _ in entries)
    panel["test_tasks"] = len(entries) - panel["calibration_tasks"]
    result = {"panel": panel, "selector": selector_audit(entries), "scenarios": {}}
    for regime in ("opaque", "disclosed"):
        reviews = disclosure if regime == "disclosed" else None
        points = [(entry[1]["split"], decision_point(entry, regime, reviews)) for entry in entries]
        menu = tuple(name for name in MENU if name in points[0][1].actions)
        test = [p for split, p in points if split == "test"]
        result["scenarios"][regime] = {
            "menu": menu,
            "sensitivity": replay([p for split, p in points if split == "cal"], test, menu,
                                  product(PENALTIES, TOKEN_PRICES, DELAY_PRICES)),
        }
        if "review" in menu:
            transitions = Counter((p.actions["stop"][-1], p.actions["review"][-1]) for p in test)
            result["scenarios"][regime]["review_failure_transitions"] = {
                str(before).lower() + "_to_" + str(after).lower(): count
                for (before, after), count in transitions.items()
            }
    if disclosure is not None:
        result["new_disclosed_reviews"] = len(disclosure)
        guarded = [(entry[1]["split"], decision_point(entry, "disclosed", disclosure, recheck_review=True))
                   for entry in entries]
        result["disclosed_with_review_recheck"] = replay(
            [p for split, p in guarded if split == "cal"], [p for split, p in guarded if split == "test"],
            MENU, [(penalty, 10, 0) for penalty in (.1, .2, .5)],
        )
    caps = read_responses(followup, "caps", plan, entries)
    if caps is not None:
        result["caps"] = cap_audit(entries, caps)
    if plan["model"] == "qwen3-8b-q4":
        note = read_responses(followup, "note", plan, entries)
        if note is not None:
            result["note"] = note_audit(entries, note)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--followup", type=Path, default=Path("results/effort_followup_2026-11-25"))
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--full-8b-selector", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("set PYTHONHASHSEED=0 to reproduce the original tasks")
    args.followup.mkdir(parents=True, exist_ok=True)
    if args.full_8b_selector:
        root = source_for(args.results, "qwen3-8b-q4")
        plan = json.loads((root / "plan.json").read_text())
        rows, coverage = load_blocks(root, plan)
        if plan["model"] != "qwen3-8b-q4" or len(rows) != 8430 or any(
            item["completed_blocks"] != item["planned_blocks"] or item["exit_code"] != 0
            for item in coverage.values()
        ):
            raise ValueError("The full selector comparison requires the completed 8B study")
        output = args.followup / "full-8b-selector.json"
        write_json(output, selector_audit(list(task_entries(rows))))
        print(output, flush=True)
        if not args.model:
            return
    for model in args.model or MODELS:
        plan, entries = load_panel(source_for(args.results, model))
        if plan["model"] != model:
            raise ValueError(f"Expected {model}, got {plan['model']}")
        report = analyze_panel(plan, entries, args.followup)
        output = args.followup / f"{model}.json"
        write_json(output, report)
        print(output, flush=True)


if __name__ == "__main__":
    main()
