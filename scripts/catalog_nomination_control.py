"""Replay card-aware checks on the frozen two-nomination catalog panels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scripts import catalog_verification as catalog
from scripts.catalog_display_control import choose as card_pick
from scripts.collect_live import write_json
from scripts.model_breadth import catalog_seed, validate_calls, without_json_fence
from scripts.model_breadth_report import MODELS


PARSERS = ("strict", "unfenced", "leading_json")


def draft_pairs(directory, public, entry):
    selected = {}
    manifests = sorted(directory.glob("*/manifest.json"))
    for path in manifests:
        data = json.loads(path.read_bytes())
        for key in ("model", "model_sha256", "frame_sha256", "corpus_sha256", "catalog_cap", "engine_revision"):
            if data[key] != entry[key]:
                raise ValueError(f"Unexpected {key} in {path}")
        for task in set(data["cases"]) & public.keys():
            if task in selected:
                raise ValueError(f"Duplicate catalog case {task} in {directory}")
            record = json.loads((path.parent / f"{task}.json").read_bytes())
            calls = record["drafts"]
            validate_calls(calls, 2, catalog_seed(task), entry["catalog_cap"], entry["model"] == "gpt-oss-20b-mxfp4")
            selected[task] = calls
    if selected.keys() != public.keys():
        raise ValueError(f"Incomplete catalog panel in {directory}")
    return selected


def replay(public, panels, drafts, parser):
    by_provider = defaultdict(lambda: defaultdict(Counter))
    for task, case in public.items():
        offers = panels[case[0]]["offers"]
        calls = drafts[task]
        if parser != "strict":
            calls = [without_json_fence(call, allow_trailing_text=parser == "leading_json") for call in calls]
        pool = [offer for call in calls if (offer := catalog.record_choice(call, offers)) is not None]
        best = catalog.pick(offers, case, True)
        policies = {"headline": catalog.pick(pool, case, False), "card": card_pick(pool, case),
                    "exact": catalog.pick(pool, case, True)}
        for regime, nominee in policies.items():
            by_provider[case[0]][regime].update(catalog.outcome(nominee, best, case), tasks=1,
                                               feasible=best is not None, referrals=nominee is None,
                                               feasible_referrals=best is not None and nominee is None)
    pooled = {regime: catalog.totals([provider[regime] for provider in by_provider.values()])
              for regime in ("headline", "card", "exact")}
    return {"pooled": pooled, "providers": by_provider}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frame", "corpus", "archive", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frame, corpus, reference = (path.read_bytes() for path in (args.frame, args.corpus, args.reference))
    panels = catalog.validate(json.loads(frame), json.loads(corpus))
    _, public = catalog.tasks(json.loads(frame), panels)
    entries = json.loads(reference)["models"]
    digests = {"frame_sha256": sha256(frame).hexdigest(), "corpus_sha256": sha256(corpus).hexdigest()}
    if len(public) != 96 or [entry["model"] for entry in entries] != list(MODELS):
        raise ValueError("Expected the full frozen catalog panel")
    models = {}
    for entry in entries:
        if any(entry[field] != value for field, value in digests.items()):
            raise ValueError(f"Wrong frame or corpus for {entry['model']}")
        drafts = draft_pairs(args.archive / entry["model"], public, entry)
        models[entry["model"]] = formats = {}
        for mode in PARSERS:
            formats[mode] = result = replay(public, panels, drafts, mode)
            prior = entry["catalogs"]["pooled"]
            if mode != "strict":
                prior = prior[mode]
            for regime, baseline in (("headline", "opaque"), ("exact", "verified")):
                if any(result["pooled"][regime][key] != value for key, value in prior[baseline].items()):
                    raise ValueError(f"Old baseline did not reproduce for {entry['model']}, {mode}, {regime}")
    pooled = {mode: {regime: catalog.totals([result[mode]["pooled"][regime] for result in models.values()])
                     for regime in ("headline", "card", "exact")} for mode in PARSERS}
    write_json(args.output, {**digests, "reference_sha256": sha256(reference).hexdigest(),
                             "models": models, "pooled": pooled})


if __name__ == "__main__":
    main()
