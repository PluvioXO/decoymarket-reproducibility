"""Give the deterministic catalog comparator the price and timing shown on each offer card."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re

from scripts.catalog_verification import outcome, pick, tasks, validate
from scripts.collect_live import write_json


AMOUNT = r"\$(\d+(?:\.\d{1,2})?)"


def cents(value):
    return int(100 * Decimal(value))


def advertised_cost(offer, months):
    # shortcut: parse the frozen card summaries, extend before scoring new billing formats.
    card = offer["display"]
    annual = re.search(AMOUNT + r" per year", card)
    if annual:
        amount = cents(annual[1])
        return amount, amount
    prepay = re.search(r"prepay " + AMOUNT + r" for the first (12|three) months", card)
    if prepay:
        term = {"12": 12, "three": 3}[prepay[2]]
        amount = cents(prepay[1])
        return (amount, amount) if months <= term else None
    first_month = re.search(AMOUNT + r" first month", card)
    initial = cents(first_month[1]) if first_month else offer["headline"]
    return initial + (months - 1) * offer["headline"], initial


def choose(offers, case):
    _, months, quantity, budget, first_bill = case
    eligible = []
    for offer in offers:
        tier = offer["advertised_quantity"]
        price = advertised_cost(offer, months)
        if (tier == "unlimited" or tier >= quantity) and price is not None and price[0] <= budget and price[1] <= first_bill:
            eligible.append((price[0], offer["id"], offer))
    return min(eligible)[-1] if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-panel", action="store_true",
                        help="Use the 96 tasks sampled for model testing, with all cards available.")
    args = parser.parse_args()
    frame = json.loads(args.frame.read_bytes())
    panels = validate(frame, json.loads(args.corpus.read_bytes()))
    all_cases, model_panel = tasks(frame, panels)
    cases = model_panel if args.model_panel else all_cases
    by_provider = defaultdict(lambda: defaultdict(Counter))
    for case in cases.values():
        offers = panels[case[0]]["offers"]
        reference = pick(offers, case, True)
        for regime, decision in (("headline", pick(offers, case, False)), ("display", choose(offers, case))):
            by_provider[case[0]][regime].update(cases=1, feasible=reference is not None, **outcome(decision, reference, case))
    write_json(args.output, by_provider)


if __name__ == "__main__":
    main()
