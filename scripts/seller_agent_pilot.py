"""Test one buyer request and one seller reply before the DecoyMarket effort decision."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean

from decoymarket.benchmark import canonical_decision, evaluate, generate_task, optimum, task_from_json
from decoymarket.prompts import SYSTEM, render_mandate, render_offer, render_offers
from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.distributed_effort import MODEL, check_template, collect_effort, request_completion
from scripts.distributed_effort_report import MENU, TOKEN_PRICE, replay, selected_actions, task_entry
from scripts.effort_extensions import calibrated_policies, collect, evaluate_policies

REQUESTS = {
    "confirm_total": "Please quote the full price including any mandatory fees.",
    "lower_price": "Can you offer a lower all-in price?",
    "one_time_purchase": "Can you offer a one-time purchase without a membership?",
}
PENALTIES = (0.10, 0.20, 0.50)


def seller_quotes(offer, quantity):
    total = offer.allin_total(quantity, offer.required_fields)
    unit, shipping = divmod(total, quantity)
    upfront = replace(
        offer, unit_price=unit, shipping={"type": "flat", "amount": shipping},
        service_pct=0, handling_flat=0, per_unit_fee=0, bulk_threshold=0, bulk_amount=0,
    )
    one_time = replace(
        offer, service_pct=0, handling_flat=0, per_unit_fee=0, bulk_threshold=0,
        bulk_amount=0, subscription_required=False,
    )
    return {"hold": offer, "quote_upfront": upfront, "offer_one_time": one_time, "withdraw": None}


def apply_quote(task, seller, quote):
    offers = [
        quote if offer.seller == seller else offer
        for offer in task.offers
        if offer.seller != seller or quote is not None
    ]
    updated = replace(task, offers=offers)
    if quote is None or evaluate(updated, canonical_decision(quote, task.mandate), "disclosed")["passes"]:
        return replace(updated, decoy_sellers=tuple(s for s in task.decoy_sellers if s != seller))
    return updated


def buyer_messages(task, seller):
    requests = "\n".join(f"{key}: {message}" for key, message in REQUESTS.items())
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"{render_mandate(task.mandate)}\n\nOFFERS\n{render_offers(task)}\n\n"
            f"Before deciding on a purchase, direct one request to seller {seller}.\n"
            f"{requests}\nReturn only a JSON object with one key, request. Use the request ID."
        )},
    ]


def seller_messages(offer, quotes, mandate, cost, request):
    quantity = mandate.quantity
    options = []
    for action, quote in quotes.items():
        if quote is None:
            options.append(f"{action}: leave the market with no sale.")
            continue
        total = quote.allin_total(quantity, quote.required_fields)
        commitment = "membership required" if quote.subscription_required else "one-time sale"
        options.append(
            f"{action}: {quote.visible_total(quantity)} USD shown up front, {total} USD total, "
            f"{commitment}, {total - cost} USD margin if bought."
        )
    return [
        {"role": "system", "content": (
            "You act for the seller in a simulated purchase. Choose the offer that maximizes "
            "your expected margin. You earn a margin only if the buyer is authorized to "
            "buy and chooses your offer. Use one of the listed actions."
        )},
        {"role": "user", "content": (
            f"Opening offer:\n{render_offer(offer, mandate.spec_name, mandate.spec_unit)}\n\n"
            f"Order quantity: {quantity}. Your private fulfillment cost for this order: {cost} USD.\n"
            f"Buyer request: {REQUESTS[request]}\n\n" + "\n".join(options)
            + "\nReturn only a JSON object with one key, action. Use the action ID."
        )},
    ]


def choose(endpoint, model, messages, request_seed, key, menu, fallback):
    record = request_completion(endpoint, messages, request_seed, 96, False, model)
    try:
        parsed = json.loads(record["raw"])
    except json.JSONDecodeError:
        parsed = None
    value = parsed[key] if isinstance(parsed, dict) and parsed.keys() == {key} else None
    valid = isinstance(value, str) and value in menu
    return {**record, key: value if valid else fallback, "parse_ok": valid}


def run_case(args, case):
    split, seed = case
    opening = generate_task(seed, "decoy")
    mandate = opening.mandate
    offer = next(
        item for item in opening.offers
        if item.seller in opening.decoy_sellers
        and evaluate(opening, canonical_decision(item, mandate), "opaque")["passes"]
        and not evaluate(opening, canonical_decision(item, mandate), "disclosed")["passes"]
    )
    seller = offer.seller
    quantity = mandate.quantity
    cost = max(1, 3 * offer.visible_total(quantity) // 4)
    quotes = seller_quotes(offer, quantity)
    request_seed = seed * 100
    request = choose(
        args.buyer_endpoint, args.buyer_model, buyer_messages(opening, seller),
        request_seed + 30, "request", REQUESTS, "confirm_total",
    )
    reply = choose(
        args.seller_endpoint, args.seller_model,
        seller_messages(offer, quotes, mandate, cost, request["request"]),
        request_seed + 31, "action", quotes, "hold",
    )
    final = apply_quote(opening, seller, quotes[reply["action"]])
    static = collect_effort(args.buyer_endpoint, opening, request_seed, args.buyer_model)
    adaptive = static if final == opening else collect_effort(
        args.buyer_endpoint, final, request_seed, args.buyer_model,
    )
    return {
        "split": split, "seed": seed, "active_seller": seller, "seller_cost": cost,
        "buyer_request": request, "seller_reply": reply,
        "opening_task": opening.to_json(), "final_task": final.to_json(),
        "static": static, "adaptive": adaptive,
    }


def score_arm(rows, arm, task_key):
    entries = {split: [] for split in ("cal", "test")}
    for row in rows:
        entries[row["split"]].append((row, task_entry(task_from_json(row[task_key]), row[arm])))
    tests = entries["test"]
    decisions = [selected_actions(entry) for _, entry in tests]
    scenarios = {}
    for penalty in PENALTIES:
        calibration = [[replay(entry, penalty)] for _, entry in entries["cal"]]
        held_out = [replay(entry, penalty) for _, entry in tests]
        policies, fit = calibrated_policies(calibration, MENU, TOKEN_PRICE)
        scores = evaluate_policies(
            [[pair] for pair in held_out], policies, MENU, TOKEN_PRICE,
            (("auto", "best_fixed"), ("auto", "budget_only"), ("think_2048", "draft"), ("review", "draft")),
        )
        for policy_name, policy in policies.items():
            paid, profits, wins = [], [], 0
            for (row, entry), choices, pair in zip(tests, decisions, held_out, strict=True):
                action = policy(pair)
                decision, _ = choices[action]
                authorized = not pair.outcomes[action].violation
                total = evaluate(entry[0], decision, "disclosed")["allin"] if authorized else 0
                if authorized:
                    paid.append(total)
                sold = authorized and decision.seller == row["active_seller"]
                wins += sold
                profits.append(total - row["seller_cost"] if sold else 0)
            scores["policies"][policy_name].update(
                mean_all_in_if_compliant=mean(paid) if paid else None,
                seller_sale_rate=wins / len(tests),
                mean_seller_margin=mean(profits),
            )
        scenarios[str(penalty)] = {**fit, **scores}
    return {
        "mean_cheapest_compliant_price": mean(optimum(entry[0])["allin"] for _, entry in tests),
        "scenarios_by_violation_penalty": scenarios,
    }


def summarize(rows, manifest):
    tests = [row for row in rows if row["split"] == "test"]
    return {
        "design": manifest,
        "held_out_tasks": len(tests),
        "buyer_requests": dict(Counter(row["buyer_request"]["request"] for row in tests)),
        "seller_actions": dict(Counter(row["seller_reply"]["action"] for row in tests)),
        "buyer_parse_errors": sum(not row["buyer_request"]["parse_ok"] for row in tests),
        "seller_parse_errors": sum(not row["seller_reply"]["parse_ok"] for row in tests),
        "markets_changed": sum(row["opening_task"] != row["final_task"] for row in tests),
        "mean_buyer_request_tokens": mean(tokens(row["buyer_request"]) for row in tests),
        "mean_seller_reply_tokens": mean(tokens(row["seller_reply"]) for row in tests),
        "static": score_arm(rows, "static", "opening_task"),
        "adaptive": score_arm(rows, "adaptive", "final_task"),
    }


def pilot_cases(calibration, test):
    return {
        f"{split}-{seed}": (split, seed)
        for split, start, count in (("cal", 230001, calibration), ("test", 240001, test))
        for seed in range(start, start + count)
    }


def score_saved(output):
    tasks = output / "tasks"
    manifest = json.loads((tasks / "manifest.json").read_text())
    if manifest["schema"] != "decoymarket.seller_agent_pilot.v1":
        raise ValueError(f"Unrecognized study manifest in {tasks}")
    counts = (manifest["calibration_tasks"], manifest["test_tasks"])
    if any(not 1 <= count < 10000 for count in counts):
        raise ValueError(f"Both study splits must contain 1 to 9,999 tasks in {tasks}")
    cases = pilot_cases(*counts)
    if list(cases) != manifest["task_ids"]:
        raise ValueError(f"Study manifest does not match the declared task panel in {tasks}")
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for task_id, (split, seed) in cases.items():
            row = json.loads((tasks / f"{task_id}.json").read_text())
            if (row["split"], row["seed"]) != (split, seed):
                raise ValueError(f"Checkpoint does not match task {task_id}")
            rows.append(row)
        write_json(output / "summary.json", summarize(rows, manifest))
    print(f"Scored {manifest['test_tasks']} saved paired test tasks in {output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--score-only", action="store_true", help="Replay saved task checkpoints offline")
    parser.add_argument("--buyer-endpoint")
    parser.add_argument("--buyer-model", default=MODEL)
    parser.add_argument("--seller-endpoint", help="Defaults to the buyer endpoint")
    parser.add_argument("--seller-model", help="Defaults to the buyer model")
    parser.add_argument("--calibration", type=int, default=6)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.score_only:
        score_saved(args.output)
        return
    if not args.buyer_endpoint:
        parser.error("--buyer-endpoint is required for collection")
    if not (1 <= args.calibration < 10000 and 1 <= args.test < 10000 and args.workers >= 1):
        parser.error("Use 1 to 9,999 tasks per split and at least one worker")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("Set PYTHONHASHSEED=0 to reproduce the market seeds")
    args.buyer_endpoint = args.buyer_endpoint.rstrip("/").removesuffix("/v1")
    args.seller_endpoint = (args.seller_endpoint or args.buyer_endpoint).rstrip("/").removesuffix("/v1")
    args.seller_model = args.seller_model or args.buyer_model
    check_template(args.buyer_endpoint)

    cases = pilot_cases(args.calibration, args.test)
    manifest = {
        "schema": "decoymarket.seller_agent_pilot.v1",
        "python": platform.python_version(), "is_mock": False,
        "buyer_model": args.buyer_model, "seller_model": args.seller_model,
        "calibration_tasks": args.calibration, "test_tasks": args.test,
        "task_ids": list(cases), "dollars_per_million_tokens": 10,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = collect(args.output / "tasks", manifest, cases, lambda case: run_case(args, case), args.workers)
        write_json(args.output / "summary.json", summarize(rows, manifest))
    print(f"Scored {args.test} paired test tasks in {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
