"""Replay price authorization with complete quotes and a scoped source lookup."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.collect_live import write_json
from scripts.public_label_extraction import ANNUAL_BUDGETS, DRAWS, MODELS, audit_budget, cost12, parse_terms


def within_both(first, second, budget):
    return first is not None and second is not None and max(first, second) <= budget


def score_draw(page, panel, table, truth):
    visible, isolated, reference, displayed = map(cost12, (page, panel, table, truth))
    if displayed is None:
        raise ValueError("The displayed label must support the budget replay")
    outcomes = defaultdict(Counter)
    for budget in ANNUAL_BUDGETS:
        requires_scope = reference is not None and (visible is None or (visible <= budget) != (reference <= budget))
        reports = {
            "page_only": (visible is not None and visible <= budget, False),
            "exact_match": (within_both(visible, reference, budget) and page[:5] == table[:5], False),
            "table_only": (reference is not None and reference <= budget, False),
            "both_full_page": (within_both(visible, reference, budget), False),
            "always_scoped": (within_both(isolated, reference, budget), reference is not None),
            "selective_scoped": (within_both(isolated if requires_scope else visible, reference, budget), requires_scope),
            "publisher_reference": (within_both(displayed, reference, budget), reference is not None),
        }
        for name, (approved, scoped) in reports.items():
            outcomes[name].update(approved=int(approved), over_display=int(approved and displayed > budget),
                                  over_either=int(approved and (displayed > budget or (reference is not None and reference > budget))),
                                  with_table=int(reference is not None), scoped_lookups=int(scoped))
    return outcomes


def replay(corpus, path, expected_digest, model):
    records = {}
    for manifest in sorted(path.glob("*/manifest.json")):
        plan = json.loads(manifest.read_text())
        if (plan["model"], plan["corpus_sha256"], plan.get("stated_units")) != (model, expected_digest, True):
            raise ValueError(f"Wrong collection for {manifest}")
        for name in plan["cases"]:
            if name in records:
                raise ValueError(f"Repeated label {name}")
            records[name] = json.loads((manifest.parent / (name + ".json")).read_text())
    if records.keys() != corpus["cases"].keys():
        raise ValueError("Incomplete public-label estimate panel")

    by_provider = defaultdict(lambda: defaultdict(Counter))
    old = Counter()
    for name, row in corpus["cases"].items():
        page, panel = records[name]["page"], records[name]["panel"]
        if len(page) != DRAWS or len(panel) != DRAWS:
            raise ValueError(f"Wrong number of calls for {name}")
        for first, second in zip(page, panel, strict=True):
            if first["seed"] != second["seed"]:
                raise ValueError(f"Unmatched extraction calls for {name}")
            estimate, isolated = (parse_terms(call["raw"], True) for call in (first, second))
            old.update(audit_budget(estimate, row["display"], row["machine"]))
            for policy, outcome in score_draw(estimate, isolated, row["machine"], row["display"]).items():
                by_provider[row["provider"]][policy].update(outcome)

    total = defaultdict(Counter)
    for provider in by_provider.values():
        for policy, outcome in provider.items():
            total[policy].update(outcome)
    for new, before in (("page_only", ""), ("exact_match", "verified_")):
        if (total[new]["approved"], total[new]["over_display"]) != (old[before + "approved"], old[before + "over_budget"]):
            raise ValueError("Prior authorization comparison did not reproduce")
    return {"rows": len(corpus["cases"]) * DRAWS * len(ANNUAL_BUDGETS), "pooled": total, "by_provider": by_provider}


def source_binding(corpus):
    by_identifier = defaultdict(list)
    same_label = []
    for name, row in corpus["cases"].items():
        display, table = map(cost12, (row["display"], row["machine"]))
        by_identifier[row["provider"], row["title"]].append({"case": name, "display": display, "table": table})
        if table is not None and table != display:
            same_label.append({"case": name, "charges": [display, table],
                               "ambiguous_budgets": [min(display, table), max(display, table)]})

    duplicate = []
    for (provider, identifier), labels in by_identifier.items():
        if len(labels) < 2:
            continue
        prices = [label["display"] for label in labels]
        lower, upper = min(prices), max(prices)
        duplicate.append({"provider": provider, "identifier": identifier, "labels": labels,
                          "ambiguous_budgets": [lower, upper],
                          "original_ceilings_in_interval": [cap for cap in ANNUAL_BUDGETS if lower <= cap < upper]})
    return {"display_table_disagreements": same_label, "duplicate_identifiers": duplicate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (args.study / "corpus.json").read_bytes()
    digest = sha256(data).hexdigest()
    corpus = json.loads(data)
    output = {model: replay(corpus, args.study / "raw-stated" / model, digest, model) for model in MODELS}
    write_json(args.output, {"corpus_sha256": digest, "models": output, "binding": source_binding(corpus)})


if __name__ == "__main__":
    main()
