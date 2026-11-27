"""Move fees into the displayed quote without changing the principal's price."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision, classify_failure, optimum, task_from_json
from decoymarket.prompts import drafter_prompt
from scripts.collect_live import write_json
from scripts.distributed_effort import ENGINE_REVISION, MODEL, MODEL_REVISION, MODEL_SHA256, check_template, complete
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked
from scripts.seller_agent_pilot import apply_quote, seller_quotes
from scripts.seller_exchange import load_rows


def prepare(rows):
    cases = {}
    for row in rows:
        if row["split"] != "test":
            continue
        opening = task_from_json(row["opening_task"])
        seller = row["active_seller"]
        offer = next(offer for offer in opening.offers if offer.seller == seller)
        quantity = opening.mandate.quantity
        quote = seller_quotes(offer, quantity)["quote_upfront"]
        quoted = apply_quote(opening, seller, quote)
        if (quote.allin_total(quantity) != offer.allin_total(quantity)
                or quote.subscription_required != offer.subscription_required or optimum(quoted) != optimum(opening)):
            raise ValueError(f"The upfront quote changed the acquisition terms on {opening.task_id}")
        quoted_prompt = drafter_prompt(quoted)
        reused = None
        if quoted_prompt == drafter_prompt(opening):
            reused = row["static"]
        elif quoted_prompt == drafter_prompt(task_from_json(row["final_task"])):
            reused = row["adaptive"]
        cases[opening.task_id] = (opening, quoted, row["static"], reused)
    return cases


def manifest(cases, shard, shards):
    requests = {key: drafter_prompt(task) for key, task in cases.items()}
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    return {"model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "engine_revision": ENGINE_REVISION, "shard": shard, "shards": shards,
            "cases": list(cases), "prompts_sha256": digest}


def collect_case(task, endpoint):
    messages = drafter_prompt(task)
    seed = task.seed * 100
    short = [complete(endpoint, messages, seed + offset, 256, False) for offset in range(2)]
    thinking = complete(endpoint, messages, seed + 10, 2048, True)
    return {"short": short, "thinking": {"2048": thinking}}


def load_records(output, cases):
    manifests = sorted(output.glob("*/manifest.json"))
    rows = {}
    for path in manifests:
        shard = int(path.parent.name)
        work = dict(list(cases.items())[shard::len(manifests)])
        files = {file.stem: file for file in path.parent.glob("*.json") if file != path}
        if (not 0 <= shard < len(manifests) or json.loads(path.read_text()) != manifest(work, shard, len(manifests))
                or set(files) != set(work) or set(files) & set(rows)):
            raise ValueError(f"Inconsistent upfront-quote shard {path.parent}")
        for key, file in files.items():
            record = json.loads(file.read_text())
            calls = [*record["short"], record["thinking"]["2048"]]
            seeds = [work[key].seed * 100 + offset for offset in (0, 1, 10)]
            if [call["seed"] for call in calls] != seeds or [call["max_tokens"] for call in calls] != [256, 256, 2048]:
                raise ValueError(f"Inconsistent upfront-quote calls in {file}")
            rows[key] = record
    if set(rows) != set(cases):
        raise ValueError("The upfront-quote panel is incomplete")
    return rows


def pool(record):
    return [Decision.from_json(call["decision"]) for call in [*record["short"][:2], record["thinking"]["2048"]]]


def summarize(cases, collected):
    if len(cases) != 144:
        raise ValueError(f"Expected 144 paired offers, got {len(cases)}")
    scores = {}
    for regime in ("opaque", "disclosed"):
        modes = {name: [] for name in ("opening", "reranked", "redrafted")}
        for key, (opening, quote, original, reused) in cases.items():
            initial = pool(original)
            updated = pool(collected[key] if reused is None else reused)
            designs = ((opening, initial), (quote, initial), (quote, updated))
            for name, (task, decisions) in zip(modes, designs, strict=True):
                outcome = classify_failure(task, ranked(task, decisions, regime))
                modes[name].append({"optimal": outcome["success"], "failure": outcome["violation"] not in ("none", "overpay")})
        scores[regime] = {}
        for outcome in ("optimal", "failure"):
            values = {name: [row[outcome] for row in rows] for name, rows in modes.items()}
            scores[regime][outcome] = {name: fmean(results) for name, results in values.items()}
            scores[regime][outcome].update({f"{name}_minus_opening": interval(values[name], values["opening"])
                                           for name in ("reranked", "redrafted")})
    return {"cases": len(cases), "new_calls_cases": len(collected), "reused_cases": len(cases) - len(collected), "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    cases = prepare(load_rows(args.exchange))
    missing = {key: quote for key, (_, quote, _, reused) in cases.items() if reused is None}
    if args.endpoint:
        if not 0 <= args.shard < args.shards <= len(missing):
            parser.error("invalid shard")
        check_template(args.endpoint)
        work = dict(list(missing.items())[args.shard::args.shards])
        collect(args.output / str(args.shard), manifest(work, args.shard, args.shards), work,
                partial(collect_case, endpoint=args.endpoint), workers=2)
        return
    write_json(args.output / "summary.json", summarize(cases, load_records(args.output, missing)))


if __name__ == "__main__":
    main()
