"""Test longer native reasoning on the same CartTrace card and receipt audits."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

from scripts.checkout_expansion import choose
from scripts.collect_live import write_json
from scripts.distributed_effort import ENGINE_REVISION, check_template, load_checkpoint, request_completion
from scripts.effort_extensions import collect
from scripts.family_transfer import MODEL as GPT_OSS, verify_template as verify_gpt_oss
from scripts.model_breadth import catalog_seed, without_json_fence


CASE_COUNT = 384
CAPS = (128, 2048)
MODELS = ("qwen3-8b-q4", "qwen3-14b-q4", GPT_OSS)


def run_case(case, endpoint, model):
    seed = catalog_seed(case["id"]) + 1
    shopper = "\n".join(case["shopper"])
    items = json.dumps(case["listing"], ensure_ascii=False)[:1400]
    record = json.dumps(case["receipt"], ensure_ascii=False)[:1100]
    system = {"role": "system", "content": "Audit a proposed order against the shopper's stated requirements. Do not buy anything."}
    prompt = (f"Shopper messages\n{shopper}\nProposed items and available catalog details\n{items}\n"
              "Confirm only if evidence supports the requested items and any stated total cap. Refer unresolved cases.")
    calls = {}
    for cap in CAPS:
        long = cap == CAPS[-1]
        kwargs = {"reasoning_effort": "high" if long else "low"} if model == GPT_OSS else None
        for arm in ("card", "receipt"):
            context = prompt + ("\nMerchant prepayment receipt, amounts in minor currency units\n" + record if arm == "receipt" else "")
            user = {"role": "user", "content": context + '\nReturn only {"offer":"approve"} or {"offer":"refer"} as the final answer.'}
            reply = request_completion(endpoint, [system, user], seed, cap, long, model, template_kwargs=kwargs)
            calls[f"{cap}_{arm}"] = {**reply, "choice": choose(without_json_fence(reply), ["approve", "refer"])}
    return {"id": case["id"], "correct": case["correct"], "calls": calls}


def summarize(cases, rows):
    score = Counter()
    for case, row in zip(cases, rows, strict=True):
        assert (case["id"], case["correct"]) == (row["id"], row["correct"])
        label = "right" if case["correct"] else "wrong"
        score[label] += 1
        for arm, call in row["calls"].items():
            score[f"{arm}_{label}_approval"] += call["choice"] == "approve"
            score[f"{arm}_unparsed"] += call["choice"] is None
            score[f"{arm}_reasoned"] += call["reasoning_characters"] > 0
            score[f"{arm}_cutoffs"] += call["finish_reason"] == "length"
            score[f"{arm}_tokens"] += call["completion_tokens"] + call["prompt_tokens"]
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint["model"]
    if model not in MODELS:
        parser.error(f"No native reasoning settings for {model}")
    (verify_gpt_oss if model == GPT_OSS else check_template)(args.endpoint)
    data = args.panel.read_bytes()
    cases = json.loads(data)["cases"][:CASE_COUNT]
    assert len(cases) == CASE_COUNT and len({case["id"] for case in cases}) == CASE_COUNT
    manifest = {"checkpoint": checkpoint, "engine_revision": ENGINE_REVISION, "panel_sha256": sha256(data).hexdigest(),
                "caps": list(CAPS), "cases": [c["id"] for c in cases]}
    rows = collect(args.output / "calls", manifest, {c["id"]: c for c in cases},
                   partial(run_case, endpoint=args.endpoint, model=model), workers=2)
    write_json(args.output / "summary.json", {"model": model, "tasks": len(cases), "scores": summarize(cases, rows)})


if __name__ == "__main__":
    main()
