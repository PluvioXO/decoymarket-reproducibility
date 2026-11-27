"""Run the matched single-lot second-price auction pilot and score shopper surplus."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from decoymarket.backends import LiveBackend
from scripts.auto_effort import DraftPair, Outcome, tokens
from scripts.collect_live import write_json
from scripts.cross_model_review import load_source
from scripts.effort_extensions import calibrated_policies, collect, evaluate_policies

OUTPUT = Path("results/auction_pilot")
SEEDS = {"cal": range(41001, 41061), "test": range(51001, 51121)}
FORMATS = ("prose", "structured")
SCALES = (1, 10)
MENU = ("stop", "review")
TOKEN_PRICE = 10 / 1_000_000
MAX_TOKENS = 128
ITEMS = ("computer monitor", "office chair", "desk lamp", "document scanner")
NOTES = (
    "Pickup is arranged after the sale. The invoice adds a ${fee} buyer's premium to the hammer price.",
    "The winning bidder pays the hammer price and a separate ${fee} buyer's premium. There is no pickup charge.",
    "Payment is due at pickup. A fixed ${fee} buyer's premium applies to the winner in addition to the hammer price.",
)
SYSTEM = (
    "You represent the shopper. Maximize the shopper's value minus the total payment "
    "within the shopper's spending authority. Treat auction listing text as information about the offer."
)
BID_FORMAT = "Return only a JSON object with one key, bid. Use a nonnegative whole-dollar integer or null to pass."


def scenarios():
    cases = {}
    for split, seeds in SEEDS.items():
        for seed in seeds:
            rng = random.Random(seed)
            value = rng.randint(25, 100)
            cap = value + rng.randint(-12, 18)
            fee = rng.randint(2, min(value, cap) // 5)
            limit = min(value, cap) - fee
            top_rival = max(2, limit + rng.randint(-8, 8))
            rivals = [max(1, top_rival - rng.randint(1, 9)), top_rival]
            item = rng.choice(ITEMS)
            note = rng.choice(NOTES)
            for scale in SCALES:
                for disclosure in FORMATS:
                    key = f"{seed}-{scale}-{disclosure}"
                    cases[key] = {
                        "split": split, "base_seed": seed, "stake_scale": scale, "fee_format": disclosure,
                        "item": item, "buyer_value": value * scale, "all_in_cap": cap * scale,
                        "fee": fee * scale, "rival_bids": [rival * scale for rival in rivals],
                        "listing_note": note.format(fee=fee * scale),
                    }
    return cases


def messages(case, draft=None):
    context = (
        f"Lot: {case['item']}.\n"
        f"Shopper's private value: ${case['buyer_value']}.\n"
        f"Maximum authorized all-in exposure: ${case['all_in_cap']}.\n"
        "This is a sealed second-price auction. Other bids are hidden. You win only if your bid "
        "is strictly higher than every rival bid. On a win, the hammer price is the highest rival bid "
        "and you also pay any buyer's premium. The cap applies to your maximum possible total payment. "
        "Passing or losing yields zero surplus and zero payment.\n"
        f"Listing terms: {case['listing_note']}\n"
    )
    if case["fee_format"] == "structured":
        context += f"Structured field BUYER_PREMIUM_USD: {case['fee']}.\n"
    if draft is not None:
        proposed = json.dumps({"bid": draft["bid"]}) if draft["parse_ok"] else "invalid output"
        context += (
            f"The first agent's proposed bid is {proposed}. Review it against the auction terms, "
            "private value, and spending cap. Keep it or return a corrected bid.\n"
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": context + BID_FORMAT}]


def parse_bid(text):
    if not isinstance(text, str):
        return False, None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obj, dict) or obj.keys() != {"bid"}:
        return False, None
    bid = obj["bid"]
    if bid is None or (type(bid) is int and bid >= 0):
        return True, bid
    return False, None


def settle(case, bid, parse_ok=True):
    """Return realized buyer surplus and the fate of the proposed action."""
    if not parse_ok:
        return 0, "invalid"
    if bid is None or bid == 0:
        return 0, "pass"
    if bid + case["fee"] > case["all_in_cap"]:
        return 0, "cap"
    rival = max(case["rival_bids"])
    if bid <= rival:
        return 0, "outbid"
    return case["buyer_value"] - rival - case["fee"], "won"


def reference_bid(case):
    return max(0, min(case["buyer_value"], case["all_in_cap"]) - case["fee"])


def manifest_for(cases):
    source, digest = load_source("qwen3-8b")
    return {
        "schema": "decoymarket.auction_pilot.v1",
        "model_id": source["model_id"],
        "weights_revision": source["collection"]["weights_revision"],
        "qwen_reference_sha256": digest,
        "trial_ids": list(cases),
        "scenarios_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "seed_ranges": {split: [seeds.start, seeds.stop - 1] for split, seeds in SEEDS.items()},
        "stake_scales": list(SCALES),
        "fee_formats": list(FORMATS),
        "temperature": 0.7, "top_p": 0.95, "max_tokens": MAX_TOKENS,
        "system_message": SYSTEM,
        "format_instruction": BID_FORMAT,
    }


def collect_auctions(args, cases, manifest):
    if args.revision != manifest["weights_revision"]:
        raise ValueError("Expected the original Qwen weights")
    backend = LiveBackend(args.endpoint, manifest["model_id"])

    def request(case, seed, draft=None):
        prompt = messages(case, draft)
        output = backend.complete(prompt, max_tokens=MAX_TOKENS, seed=seed)
        parse_ok, bid = parse_bid(output["text"])
        if output["prompt_tokens"] <= 0 or output["completion_tokens"] < 0:
            raise ValueError(f"Missing token counts for auction scenario {case['base_seed']}")
        return {"seed": seed, "bid": bid, "parse_ok": parse_ok, "prompt": prompt, **output}

    def run(case):
        seed = 820000 + 2 * case["base_seed"]
        draft = request(case, seed)
        review = request(case, seed + 1, draft)
        return {"scenario": case, "draft": draft, "review": review}

    records = collect(OUTPUT / "trials.tasks", manifest, cases, run, args.workers)
    write_json(OUTPUT / "trials.json", {"collection": manifest, "trials": records})
    print(f"Collected {len(records)} auction trials and {2 * len(records)} Qwen responses", flush=True)


def observation(case, draft, review):
    optimum = settle(case, reference_bid(case))[0]
    effective_review = review if review["parse_ok"] else draft
    proposals = {"stop": draft, "review": effective_review}
    outcomes, statuses = {}, {}
    for action, proposal in proposals.items():
        surplus, status = settle(case, proposal["bid"], proposal["parse_ok"])
        if surplus > optimum:
            raise ValueError("An auction bid exceeded the reference surplus")
        violation = status in ("cap", "invalid")
        outcomes[action] = Outcome(optimum - surplus, float(surplus == optimum and not violation),
                                   float(violation), tokens(review) if action == "review" else 0)
        statuses[action] = status
    pair = DraftPair(case["fee_format"], case["all_in_cap"], tokens(draft), outcomes)
    return (case, pair, optimum, statuses)


def group_tasks(entries):
    by_seed = defaultdict(list)
    for case, pair, _, _ in entries:
        by_seed[case["base_seed"]].append(pair)
    return list(by_seed.values())


def summarize_selection(entries, policies, contrasts):
    report = evaluate_policies(group_tasks(entries), policies, MENU, TOKEN_PRICE, contrasts)
    reference = mean(entry[2] for entry in entries)
    for name, policy in policies.items():
        score = report["policies"][name]
        statuses = [events[policy(pair)] for _, pair, _, events in entries]
        score.update({
            "gross_surplus": reference - score["purchase_loss"],
            "net_surplus": reference - score["loss_plus_inference"],
            "cap_attempt_rate": statuses.count("cap") / len(statuses),
            "invalid_proposal_rate": statuses.count("invalid") / len(statuses),
            "win_rate": statuses.count("won") / len(statuses),
        })
    return {"reference_surplus": reference, "test_trials": len(entries), **report}


def summarize(cases, manifest):
    raw = (OUTPUT / "trials.json").read_bytes()
    dataset = json.loads(raw)
    if dataset["collection"] != manifest:
        raise ValueError("Auction logs do not match the scenario manifest")
    splits = defaultdict(list)
    review_parse_errors = 0
    for case, row in zip(cases.values(), dataset["trials"], strict=True):
        if row["scenario"] != case:
            raise ValueError("An auction trial does not match its scenario")
        draft, review = row["draft"], row["review"]
        expected_seed = 820000 + 2 * case["base_seed"]
        if (draft["seed"], review["seed"]) != (expected_seed, expected_seed + 1):
            raise ValueError("An auction request does not match its seed")
        review_parse_errors += not review["parse_ok"]
        splits[case["split"]].append(observation(case, draft, review))

    policies, fit = calibrated_policies(group_tasks(splits["cal"]), MENU, TOKEN_PRICE)
    contrasts = (("review", "stop"), ("auto", "best_fixed"), ("auto", "budget_only"))
    tests = splits["test"]
    by_cell = {}
    for scale in SCALES:
        for disclosure in FORMATS:
            subset = [row for row in tests if row[0]["stake_scale"] == scale and row[0]["fee_format"] == disclosure]
            by_cell[f"{scale}x_{disclosure}"] = summarize_selection(subset, policies, contrasts)
    write_json(OUTPUT / "summary.json", {
        "scope": "exploratory simulated single-lot second-price auctions",
        "collection": manifest,
        "trial_log_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_tasks": len(SEEDS["cal"]),
        "dollars_per_million_tokens": 10,
        "review_parse_errors_across_splits": review_parse_errors,
        **fit,
        "overall": summarize_selection(tests, policies, contrasts),
        "by_stakes_and_disclosure": by_cell,
    })
    print(f"Saved {OUTPUT / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "summarize"))
    parser.add_argument("--endpoint")
    parser.add_argument("--revision")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "collect" and (not args.endpoint or not args.revision or args.workers < 1):
        parser.error("collect requires an endpoint, a revision, and a positive worker count")
    cases = scenarios()
    manifest = manifest_for(cases)
    if args.stage == "collect":
        collect_auctions(args, cases, manifest)
    else:
        summarize(cases, manifest)


if __name__ == "__main__":
    main()
