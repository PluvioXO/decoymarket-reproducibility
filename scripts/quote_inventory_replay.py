"""Choose from all indexed broadband records using archived stated-unit extractions."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from scripts.auto_effort import tokens
from scripts.collect_live import write_json
from scripts.public_label_extraction import ANNUAL_BUDGETS, CAP, CONDITIONS, DRAWS, MODELS, cases_for, parse_terms
from scripts.quoted_page_choice import SPEEDS, choice_metrics, eligible, index, winner


def replay(source):
    content = (source / "corpus.json").read_bytes()
    digest = sha256(content).hexdigest()
    corpus = json.loads(content)
    _, panels = index(corpus)
    keys = {(row["provider"], row["ordinal"]): key for key, row in corpus["cases"].items()}
    estimates = cases_for(corpus)
    result = {}
    for model in MODELS:
        predictions = {}
        token_counts = {arm: [0] * DRAWS for arm in CONDITIONS}
        for path in (source / "raw-stated" / model).glob("*/manifest.json"):
            spec = json.loads(path.read_text())
            if (spec["corpus_sha256"], spec["model"], spec["stated_units"]) != (digest, model, True):
                raise ValueError(f"Wrong corpus/model in {path}")
            for key in spec["cases"]:
                calls = json.loads((path.parent / f"{key}.json").read_text())
                seed = estimates[key][1]
                expected = [(seed + draw, CAP, False) for draw in range(DRAWS)]
                if key in predictions or set(calls) != set(CONDITIONS) or any(
                    [(row["seed"], row["max_tokens"], row["thinking"]) for row in arm] != expected for arm in calls.values()
                ):
                    raise ValueError(f"Duplicate or invalid calls at {path.parent / key}")
                predictions[key] = {arm: [parse_terms(row["raw"], stated_units=True) for row in items]
                                    for arm, items in calls.items()}
                for arm, items in calls.items():
                    for draw, row in enumerate(items):
                        token_counts[arm][draw] += tokens(row)
        if predictions.keys() != corpus["cases"].keys():
            raise ValueError(f"Missing archived panel extraction for {model}")

        outcomes = {arm: {"all": Counter(), "verified_pick": Counter()} for arm in CONDITIONS}
        for provider, rows in panels.items():
            exact = {locator: record["display"] for locator, record in rows.items()}
            for arm, draw, speed, budget in product(CONDITIONS, range(DRAWS), SPEEDS, ANNUAL_BUDGETS):
                oracle = winner(exact.items(), speed, budget)
                pool = [(locator, predictions[keys[provider, row["ordinal"]]][arm][draw]) for locator, row in rows.items()]
                pick = winner(pool, speed, budget)
                outcomes[arm]["all"].update(choice_metrics(pick, exact, oracle, speed, budget), cases=1, feasible=bool(oracle))
                verified = pick if eligible(exact.get(pick), speed, budget) else 0
                outcomes[arm]["verified_pick"].update(choice_metrics(verified, exact, oracle, speed, budget), cases=1,
                                                      feasible=bool(oracle))
        result[model] = {"calls_per_draw_per_context": len(keys), "draws": DRAWS, "tokens_per_context_and_draw": token_counts,
                         "outcomes": outcomes}
    return {"corpus_sha256": digest, "models": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, replay(args.source))


if __name__ == "__main__":
    main()
