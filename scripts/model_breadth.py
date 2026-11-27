"""Collect and score matched synthetic drafts and published catalog nominations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

from decoymarket.benchmark import Decision, generate_task
from decoymarket.prompts import drafter_prompt
from scripts import catalog_verification as catalog
from scripts.collect_live import write_json
from scripts.distributed_effort import CHECKPOINT_KEYS, ENGINE_REVISION, cases_for_block, check_template
from scripts.distributed_effort import complete, load_checkpoint, request_completion
from scripts.draft_family_expansion import CAP, DRAFTS, ENCODINGS, seed_for, verify_template, with_encoding
from scripts.effort_extensions import collect
from scripts.effort_followup_report import interval, ranked
from scripts.family_transfer import MODEL as GPT_OSS, verify_template as verify_reasoning
from scripts.unconditioned_fees import SEEDS, cases as fee_cases, outcome


def synthetic_tasks():
    tasks = {}
    for block in range(12):
        for market, seed, split in cases_for_block(block):
            if split == "test":
                task = generate_task(seed, market)
                tasks[task.task_id] = (market, task)
    tasks.update((name, ("unconditioned", task)) for name, task in fee_cases(SEEDS).items() if name.endswith("-none"))
    return tasks


def caps(model):
    return (768, 768) if model == GPT_OSS else (CAP, catalog.CAP)


def collect_case(item, endpoint, model, encoding, synthetic, public, panels):
    kwargs = {"reasoning_effort": "low"} if model == GPT_OSS else None
    short_cap, catalog_cap = caps(model)

    if item in synthetic:
        _, task = synthetic[item]
        messages = with_encoding(drafter_prompt(task), encoding)
        return [complete(endpoint, messages, seed_for(task) + offset, short_cap, model == GPT_OSS, model,
                         template_kwargs=kwargs) for offset in range(DRAFTS)]

    case = public[item]
    panel = panels[case[0]]
    seed = catalog_seed(item)

    def nominate(offset, evidence=None):
        messages = with_encoding(catalog.prompt(case, panel, evidence), encoding)
        return request_completion(endpoint, messages, seed + offset, catalog_cap, model == GPT_OSS, model,
                                  template_kwargs=kwargs)

    drafts = [nominate(0), nominate(1)]
    first = catalog.source(catalog.record_choice(drafts[0], panel["offers"]))
    evidence = catalog.disclosed(panel["offers"], first)
    return {"drafts": drafts, "with_first_source": nominate(2, evidence)}


def catalog_seed(key):
    return int.from_bytes(sha256(key.encode()).digest()[:4], "big") >> 2


def validate_calls(calls, count, seed, cap, thinking):
    expected = [(seed + index, cap, thinking) for index in range(count)]
    actual = [(call["seed"], call["max_tokens"], call["thinking"]) for call in calls]
    if expected != actual:
        raise ValueError(f"Unexpected calls for request seed {seed}")


def selection(task, calls):
    drafts = [Decision.from_json(call["decision"]) for call in calls]
    one = [outcome(task, draft)[0] for draft in drafts]
    opaque, disclosed = (outcome(task, ranked(task, drafts, policy)) for policy in ("opaque", "disclosed"))
    return {"one_optimal": fmean(one), "pool_optimal": any(one), "opaque_optimal": opaque[0], "disclosed_optimal": disclosed[0],
            "opaque_failure": opaque[1], "disclosed_failure": disclosed[1], "recovered": not opaque[0] and disclosed[0],
            "lost": opaque[0] and not disclosed[0], "parse": fmean(call["parse_ok"] for call in calls),
            "cutoff": fmean(call["finish_reason"] == "length" for call in calls),
            "traced_reasoning": fmean(call["reasoning_characters"] > 0 for call in calls)}


def summarize_synthetic(synthetic, records, model):
    groups = defaultdict(list)
    for name, (market, task) in synthetic.items():
        calls = records[name]
        validate_calls(calls, DRAFTS, seed_for(task), caps(model)[0], model == GPT_OSS)
        groups[market].append(selection(task, calls))
    out = {}
    for market, results in groups.items():
        counts = catalog.totals(results)
        tasks = len(results)
        out[market] = {"tasks": tasks, "counts": counts, "rates": {key: value / tasks for key, value in counts.items()},
                       "disclosure_optimal": interval([r["disclosed_optimal"] for r in results], [r["opaque_optimal"] for r in results]),
                       "disclosure_failure": interval([r["disclosed_failure"] for r in results], [r["opaque_failure"] for r in results]),
                       "scale_optimal": interval([r["opaque_optimal"] for r in results], [r["one_optimal"] for r in results])}
    return out


def without_json_fence(call, *, allow_trailing_text=False):
    lines = call["raw"].strip().splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json"):
        return call
    boundaries = [index for index, line in enumerate(lines[1:], 1) if line.lstrip().startswith("```")]
    if len(boundaries) != 1 or lines[boundaries[0]].strip() != "```":
        return call
    end = boundaries[0]
    if not allow_trailing_text and end != len(lines) - 1:
        return call
    return {**call, "raw": "\n".join(lines[1:end])}


def summarize_catalogs(public, panels, records, model):
    groups = defaultdict(list)
    for key, case in public.items():
        record = records[key]
        calls = [*record["drafts"], record["with_first_source"]]
        validate_calls(calls, 3, catalog_seed(key), caps(model)[1], model == GPT_OSS)
        offers = panels[case[0]]["offers"]
        row = catalog.evaluate(case, offers, record)
        for parser, allow_trailing in (("unfenced", False), ("leading_json", True)):
            plain_drafts = [without_json_fence(call, allow_trailing_text=allow_trailing) for call in record["drafts"]]
            row[parser] = row["candidates"]
            if plain_drafts != record["drafts"]:
                row[parser] = catalog.evaluate(case, offers, {**record, "drafts": plain_drafts})["candidates"]
        row["cutoffs"] = sum(call["finish_reason"] == "length" for call in calls)
        row["reasoning_traces"] = sum(call["reasoning_characters"] > 0 for call in calls)
        groups[case[0]].append(row)

    def candidate_totals(candidates):
        return {"pool_optimal": sum(row["contains_optimum"] for row in candidates),
                "known_initial_ids": sum(row["valid_first"] + row["valid_second"] for row in candidates),
                "opaque": catalog.totals([row["opaque"] for row in candidates]),
                "verified": catalog.totals([row["verified"] for row in candidates]),
                "retained_compliant": sum(row["opaque"]["compliant"] and row["verified"]["compliant"] for row in candidates)}

    def aggregate(rows):
        return {"tasks": len(rows), "feasible": sum(row["feasible"] for row in rows),
                **candidate_totals([row["candidates"] for row in rows]),
                "unfenced": candidate_totals([row["unfenced"] for row in rows]),
                "leading_json": candidate_totals([row["leading_json"] for row in rows]),
                "cutoffs": sum(row["cutoffs"] for row in rows),
                "reasoning_traces": sum(row["reasoning_traces"] for row in rows),
                "policies": {arm: catalog.totals([row["policies"][arm] for row in rows]) for arm in catalog.POLICIES}}

    return {"pooled": aggregate([row for rows in groups.values() for row in rows]),
            "providers": {name: aggregate(rows) for name, rows in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("frame", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    frame, corpus = (path.read_bytes() for path in (args.frame, args.corpus))
    panels = catalog.validate(json.loads(frame), json.loads(corpus))
    _, public = catalog.tasks(json.loads(frame), panels)
    synthetic = synthetic_tasks()
    if (len(synthetic), len(public), len(panels)) != (416, 96, 8) or public.keys() & synthetic.keys():
        raise ValueError("The task grid does not match the frozen synthetic and catalog panels")
    cases = list(synthetic) + list(public)
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint["model"]
    settings = {**{key: checkpoint[key] for key in CHECKPOINT_KEYS}, "engine_revision": ENGINE_REVISION,
                "frame_sha256": sha256(frame).hexdigest(), "corpus_sha256": sha256(corpus).hexdigest(),
                "synthetic_cap": caps(model)[0], "catalog_cap": caps(model)[1], "mode": "low" if model == GPT_OSS else "short"}

    def manifest(shard, shards, encoding):
        return {**settings, "shard": shard, "shards": shards, "encoding": encoding, "cases": cases[shard::shards]}

    if args.endpoint:
        if not 0 <= args.shard < args.shards <= len(cases):
            parser.error("invalid shard")
        if model == GPT_OSS:
            verify_reasoning(args.endpoint)
            encoding = "native"
        else:
            if model.startswith("qwen3-"):
                check_template(args.endpoint)
            encoding = verify_template(args.endpoint, next(iter(synthetic.values()))[1])
        spec = manifest(args.shard, args.shards, encoding)
        collect(args.output / str(args.shard), spec, {name: name for name in spec["cases"]},
                partial(collect_case, endpoint=args.endpoint, model=model, encoding=encoding,
                        synthetic=synthetic, public=public, panels=panels), workers=2)
        return

    records, encodings = {}, set()
    paths = sorted(args.output.glob("*/manifest.json"))
    for path in paths:
        saved = json.loads(path.read_text())
        shard, encoding = saved["shard"], saved["encoding"]
        if (not 0 <= shard < len(paths) or path.parent.name != str(shard) or encoding not in ENCODINGS
                or saved != manifest(shard, len(paths), encoding) or set(saved["cases"]) & records.keys()
                or {p.stem for p in path.parent.glob("*.json") if p != path} != set(saved["cases"])):
            raise ValueError(f"Incomplete or incompatible shard {path.parent}")
        encodings.add(encoding)
        records.update((key, json.loads((path.parent / f"{key}.json").read_text())) for key in saved["cases"])
    if set(cases) != records.keys() or len(encodings) != 1:
        raise ValueError(f"Incomplete or mixed-encoding panel for {model}")
    calls = [call for key in synthetic for call in records[key]]
    calls.extend(call for key in public for call in [*records[key]["drafts"], records[key]["with_first_source"]])
    largest_request = max(call["prompt_tokens"] + call["max_tokens"] for call in calls)
    slot_capacity = 2048 if model == "olmo3-7b-q4" else 4096
    if largest_request > slot_capacity:
        raise ValueError(f"A prompt and its completion cap exceeded the loaded context for {model}")
    write_json(args.output / "summary.json", {**settings, "encoding": encodings.pop(),
                                              "largest_request": largest_request, "slot_capacity": slot_capacity,
                                              "synthetic": summarize_synthetic(synthetic, records, model),
                                              "catalogs": summarize_catalogs(public, panels, records, model)})


if __name__ == "__main__":
    main()
