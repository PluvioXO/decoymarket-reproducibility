"""Repeat full-page quote nominations with native reasoning and unguided JSON."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS, ENGINE_REVISION, check_template, load_checkpoint, request_completion
from scripts.effort_extensions import collect
from scripts.family_transfer import MODEL as GPT_OSS, verify_template
from scripts.quoted_page_choice import MODELS, case_grid, choice_metrics, index, messages, read_choice, winner


CAP = 4096


def run_case(case, endpoint, model, pages, level, numeric_example):
    provider, speed, budget, seed = case
    kwargs = {"reasoning_effort": level} if model == GPT_OSS else None
    prompt = messages(speed, budget, pages[provider])
    if numeric_example:
        prompt[0]["content"] += (" The first JSON item is the panel number as a bare integer. If the panel marker "
                                 "is QUOTE 4, begin the response with [4, and follow the comma with the fields array. "
                                 "Return plain JSON.")
    return request_completion(endpoint, prompt, seed, CAP, True, model, template_kwargs=kwargs)


def normalize_syntax(raw):
    raw = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced[1]
    separate = re.fullmatch(r"\[\s*QUOTE\s+(\d+)\s*\]\s*,\s*(\[.*\])", raw, flags=re.IGNORECASE | re.DOTALL)
    if separate:
        return f"[{separate[1]},{separate[2]}]"
    return re.sub(r"^\[\s*QUOTE\s+(\d+)\s*,", r"[\1,", raw, flags=re.IGNORECASE)


def summarize(cases, panels, rows):
    totals = Counter()
    outcomes = {arm: Counter() for arm in ("page", "exact", "repaired_page", "repaired_exact")}
    for key, (provider, speed, budget, seed) in cases.items():
        call = rows[key]
        if (call["seed"], call["max_tokens"], call["thinking"]) != (seed, CAP, True):
            raise ValueError(f"Unexpected reasoning settings for {key}")
        exact = {number: row["display"] for number, row in panels[provider].items()}
        oracle = winner(exact.items(), speed, budget)
        locator, terms = read_choice(call)
        repaired = read_choice({"raw": normalize_syntax(call["raw"])})
        no_quote = re.fullmatch(r"\[\s*0\s*,\s*null\s*\]", call["raw"].strip()) is not None
        totals.update(cases=1, feasible=bool(oracle), known_locator=locator in exact,
                      reasoning_trace=call["reasoning_characters"] > 0, cutoffs=call["finish_reason"] == "length",
                      explicit_abstention=no_quote, malformed=terms is None and not no_quote,
                      completion_tokens=call["completion_tokens"], prompt_tokens=call["prompt_tokens"])
        for prefix, (nominee, fields) in (("", (locator, terms)), ("repaired_", repaired)):
            outcomes[prefix + "page"].update(choice_metrics(winner([(nominee, fields)], speed, budget), exact,
                                                             oracle, speed, budget))
            outcomes[prefix + "exact"].update(choice_metrics(winner([(nominee, exact.get(nominee))], speed, budget),
                                                              exact, oracle, speed, budget))
    return {"total": totals, **outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--level", choices=("low", "high"), default="high")
    parser.add_argument("--numeric-example", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    content = args.corpus.read_bytes()
    pages, panels = index(json.loads(content))
    cases = case_grid(pages)
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint["model"]
    if model not in MODELS or (model != GPT_OSS and args.level != "high"):
        parser.error("only gpt-oss supports both effort levels")
    if args.numeric_example and (model != GPT_OSS or args.level != "low"):
        parser.error("the numeric example control uses gpt-oss low")
    settings = {**{key: checkpoint[key] for key in CHECKPOINT_KEYS}, "engine_revision": ENGINE_REVISION,
                "corpus_sha256": sha256(content).hexdigest(), "level": args.level, "cap": CAP}
    if args.numeric_example:
        settings["numeric_example"] = True

    def manifest(shard, shards):
        return {**settings, "shard": shard, "shards": shards, "cases": list(cases)[shard::shards]}

    if args.endpoint:
        if not 0 <= args.shard < args.shards <= len(cases):
            parser.error("invalid shard")
        (verify_template if model == GPT_OSS else check_template)(args.endpoint)
        spec = manifest(args.shard, args.shards)
        work = {name: cases[name] for name in spec["cases"]}
        collect(args.output / str(args.shard), spec, work,
                partial(run_case, endpoint=args.endpoint, model=model, pages=pages, level=args.level,
                        numeric_example=args.numeric_example), workers=2)
        return
    rows = {}
    manifests = list(args.output.glob("*/manifest.json"))
    for path in manifests:
        saved = json.loads(path.read_text())
        if saved != manifest(int(path.parent.name), len(manifests)):
            raise ValueError(f"Invalid settings in {path}")
        rows.update((name, json.loads((path.parent / f"{name}.json").read_text())) for name in saved["cases"])
    if rows.keys() != cases.keys():
        raise ValueError("Missing native-reasoning choices")
    write_json(args.output / "summary.json", {**settings, **summarize(cases, panels, rows)})


if __name__ == "__main__":
    main()
