"""Replay an economic stop, draft, or review decision after two proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from decoymarket.analysis import paired_delta_ci
from decoymarket.benchmark import MARKETS, Decision, classify_failure, selection_key, task_from_json

MODELS = ("qwen3-8b", "llama31-8b")
ACTIONS = ("stop", "draft", "review")
STAKE_SCALES = (0.001, 0.01, 0.1, 1.0)
REGIME = "opaque"


@dataclass(frozen=True, slots=True)
class Outcome:
    loss: float
    optimal: float
    violation: float
    extra_tokens: float


@dataclass(frozen=True, slots=True)
class DraftPair:
    state: str
    budget: int
    initial_tokens: int
    outcomes: dict[str, Outcome]


def tokens(record):
    return record["prompt_tokens"] + record["completion_tokens"]


def scored_outcome(task, decision, optimal_price, penalty, extra_tokens=0):
    result = classify_failure(task, decision)
    violation = result["violation"] not in ("none", "overpay")
    loss = optimal_price * penalty if violation else (result["overpay"] or 0)
    return Outcome(loss, float(result["success"]), float(violation), extra_tokens)


def replay_task(row, penalty, cross_reviews=None):
    task = task_from_json(row["task"])
    price = row["optimum"]["allin"]
    records = row["drafts"]
    drafts = [Decision.from_json(item["decision"]) for item in records]
    draft_tokens = [tokens(item) for item in records]
    scores = [scored_outcome(task, draft, price, penalty) for draft in drafts]
    reviews = [row["reviews"][REGIME][item["key"]] for item in records]
    reviewed = [
        scored_outcome(task, Decision.from_json(item["decision"]), price, penalty, tokens(item))
        for item in reviews
    ]
    cross_reviewed = {
        key: scored_outcome(task, Decision.from_json(item["decision"]), price, penalty, tokens(item))
        for key, item in (cross_reviews or {}).items()
    }
    ranks = [selection_key(task, draft, REGIME) + (i,) for i, draft in enumerate(drafts)]
    indices = range(len(drafts))
    total_tokens = sum(draft_tokens)
    for a, b in combinations(indices, 2):
        pick = a if ranks[a] < ranks[b] else b
        if ranks[pick][0]:
            state = "failed_check"
        else:
            state = "agreement" if drafts[a] == drafts[b] else "disagreement"
        after_draft = [
            scores[i if ranks[i] < ranks[pick] else pick]
            for i in indices if i != a and i != b
        ]
        initial = draft_tokens[a] + draft_tokens[b]
        count = len(after_draft)
        outcomes = {
            "stop": scores[pick],
            "draft": Outcome(
                sum(outcome.loss for outcome in after_draft) / count,
                sum(outcome.optimal for outcome in after_draft) / count,
                sum(outcome.violation for outcome in after_draft) / count,
                (total_tokens - initial) / count,
            ),
            "review": reviewed[pick],
        }
        if cross_reviews is not None:
            outcomes["cross_review"] = cross_reviewed[records[pick]["key"]]
        yield DraftPair(state, task.mandate.budget, initial, outcomes)


def load_split(model, split, penalty, directory, hashes):
    tasks = []
    seeds = list(range(1, 101) if split == "cal" else range(10001, 10201))
    for market in MARKETS:
        path = directory / model / f"{market}_{split}.json"
        data = path.read_bytes()
        episode = json.loads(data)
        rows = episode["tasks"]
        if (
            episode["backend"] != "live"
            or episode["is_mock"]
            or episode["stopped_early"]
            or episode["model_name"] != model
            or episode["market"] != market
            or [row["task"]["seed"] for row in rows] != seeds
            or any(len(row["drafts"]) != 16 for row in rows)
        ):
            raise ValueError(f"Expected the complete live split in {path}")
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        tasks.extend(list(replay_task(row, penalty)) for row in rows)
    return tasks


def fit(calibration, actions=ACTIONS):
    groups = defaultdict(list)
    for pairs in calibration:
        for pair in pairs:
            groups[pair.state].append(pair)
            groups["all"].append(pair)
    return {
        state: {
            action: {
                "loss_per_budget_dollar": mean(pair.outcomes[action].loss / pair.budget for pair in pairs),
                "extra_tokens": mean(pair.outcomes[action].extra_tokens for pair in pairs),
            }
            for action in actions
        }
        for state, pairs in groups.items()
    }


def auto_choice(estimates, budget, token_price, actions=ACTIONS):
    return min(actions, key=lambda action: (
        budget * estimates[action]["loss_per_budget_dollar"]
        + token_price * estimates[action]["extra_tokens"]
    ))


def assess(tasks, choose, stake_scale, token_price, actions=ACTIONS):
    counts = Counter()
    losses, accuracy, violations, total_tokens = [], [], [], []
    for pairs in tasks:
        loss = optimal = violation = spent = 0.0
        for pair in pairs:
            action = choose(pair)
            counts[action] += 1
            outcome = pair.outcomes[action]
            loss += outcome.loss * stake_scale
            optimal += outcome.optimal
            violation += outcome.violation
            spent += pair.initial_tokens + outcome.extra_tokens
        count = len(pairs)
        losses.append(loss / count)
        accuracy.append(optimal / count)
        violations.append(violation / count)
        total_tokens.append(spent / count)
    cost = [loss + token_price * spent for loss, spent in zip(losses, total_tokens)]
    chosen = sum(counts.values())
    return {
        "loss_plus_inference": mean(cost),
        "purchase_loss": mean(losses),
        "optimal_rate": mean(accuracy),
        "violation_rate": mean(violations),
        "mean_tokens": mean(total_tokens),
        "action_shares": {action: counts[action] / chosen for action in actions},
    }, cost


def compare(calibration, tests, estimates, stake_scale, token_price):
    best = min(ACTIONS, key=lambda action: assess(
        calibration, lambda pair: action, stake_scale, token_price,
    )[0]["loss_plus_inference"])
    policies = {
        "auto": lambda pair: auto_choice(
            estimates.get(pair.state, estimates["all"]), pair.budget * stake_scale, token_price,
        ),
        "price_only": lambda pair: auto_choice(estimates["all"], pair.budget * stake_scale, token_price),
        "agree2": lambda pair: "stop" if pair.state == "agreement" else "review",
        "best_fixed": lambda pair: best,
    }
    scores, per_task_cost = {}, {}
    for name, choose in policies.items():
        scores[name], per_task_cost[name] = assess(tests, choose, stake_scale, token_price)
    scores["best_fixed"]["action"] = best
    result = {"stake_scale": stake_scale, "policies": scores}
    if stake_scale == 1:
        result["auto_savings_with_paired_95_ci"] = {
            baseline: paired_delta_ci(per_task_cost[baseline], per_task_cost["auto"])
            for baseline in ("agree2", "best_fixed", "price_only")
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/auto_effort.json"))
    parser.add_argument("--violation-penalty", type=float, default=0.20)
    parser.add_argument("--dollars-per-million-tokens", type=float, default=10.0)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0
           for value in (args.violation_penalty, args.dollars_per_million_tokens)):
        parser.error("penalty and token price must be finite and nonnegative")
    token_price = args.dollars_per_million_tokens / 1_000_000
    hashes, models = {}, {}
    for model in MODELS:
        calibration = load_split(model, "cal", args.violation_penalty, args.results, hashes)
        estimates = fit(calibration)
        tests = load_split(model, "test", args.violation_penalty, args.results, hashes)
        models[model] = {
            "calibration_tasks": len(calibration),
            "test_tasks": len(tests),
            "calibration_estimates": estimates,
            "scenarios": [compare(calibration, tests, estimates, scale, token_price) for scale in STAKE_SCALES],
        }
        print(f"{model}: {len(tests)} held-out tasks replayed", flush=True)
    report = {
        "scope": "exploratory offline replay",
        "regime": REGIME,
        "initial_drafts": 2,
        "violation_penalty_fraction_of_optimal_price": args.violation_penalty,
        "dollars_per_million_tokens": args.dollars_per_million_tokens,
        "input_sha256": hashes,
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
