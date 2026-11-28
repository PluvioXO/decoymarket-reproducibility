"""Rescore the USD storefront panel with the two initial nominations fixed."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.checkout_evidence import FRAME, storefront_cases
from scripts.checkout_expansion import ranked
from scripts.checkout_expansion_report import MODELS
from scripts.collect_live import write_json


def report(root):
    frame = FRAME.read_bytes()
    source = (root / "storefronts.json").read_bytes()
    captured = json.loads(source)
    if captured["protocol_sha256"] != sha256(frame).hexdigest() or not captured["usd_confirmation"]:
        raise ValueError("Use the frozen USD storefront panel")
    offers = captured["offers"]
    tasks = {row["key"]: row for row in storefront_cases(json.loads(frame), offers, confirm_currency=True)}
    scopes = {}
    for case in tasks.values():
        cards, quotes = case["cards"], case["quotes"]
        if len(quotes) < 2:
            continue
        visible = min(quotes, key=lambda key: (cards[key][0], key))
        scopes[(case["group"], case["destination"], case["mandate"][1])] = quotes[visible][0] > min(price[0] for price in quotes.values())
    coverage = Counter(q["stage"] for o in offers for q in o["quotes"].values())
    models = {}
    for model in MODELS:
        directory = root / "checkout-retail-updated" / model / "0"
        path = directory / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        manifest = json.loads((directory / "cart/manifest.json").read_text())
        if summary["model"] != model or manifest["cart"] != sha256(source).hexdigest() or list(tasks) != manifest["tasks"]:
            raise ValueError(f"Mismatched task panel for {model}")
        details = Counter()
        for key, task in tasks.items():
            row = json.loads((directory / "cart" / f"{key}.json").read_text())
            assert row["key"] == key
            nominees = {r["choice"] for r in row["drafts"][:2]} - {None, "refer"}
            first = ranked(task, nominees, task["cards"])
            second = ranked(task, nominees - {first}, task["cards"])
            quotes = task["quotes"]
            exact = ranked(task, [first, second], quotes)
            initial = ranked(task, [first], quotes)
            details.update(distinct_second_nominee=second is not None, fixed_nominees_approved=exact is not None,
                           gains_vs_one_source=exact is not None and initial is None)
        models[model] = {"all": summary["cart"]["all"], "measured_source": summary["cart"].get("measured_source", {}),
                         "fixed_nominations": details}
    return {"tasks_per_model": len(tasks), "quote_scopes": coverage, "comparable_scopes": len(scopes),
            "price_reversals": sum(scopes.values()), "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, report(args.run_root))


if __name__ == "__main__":
    main()
