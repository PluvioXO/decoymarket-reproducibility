"""Collect and schedule a matched DecoyMarket inference-budget study."""
from __future__ import annotations

import argparse
import json
import math
import re
import signal
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decoymarket.backends import parse_decision
from decoymarket.benchmark import Decision, MARKETS, evaluate, generate_task, select
from decoymarket.prompts import drafter_prompt, reviewer_prompt
from decoymarket.run import INVALID
from scripts.collect_live import write_json

MODEL = "qwen3-8b-q4"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
ENGINE_REVISION = "3de008208b9b8a33f49f979097a99b4d59e6e521"
SHORT_DRAFTS = 8
THINK_CAPS = (1024, 2048)
TASKS_PER_BLOCK = 10 * len(MARKETS)
HOURS = 12
CHECKPOINT_KEYS = ("model", "model_revision", "model_sha256")


def load_checkpoint(root):
    manifest = root / "model" / "checkpoint.json"
    if not manifest.exists():
        return {
            "model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "weights": "Qwen3-8B-Q4_K_M.gguf",
        }
    checkpoint = json.loads(manifest.read_text())
    patterns = {
        "model": r"[a-z0-9][a-z0-9-]*",
        "model_revision": r"[0-9a-f]{40}",
        "model_sha256": r"[0-9a-f]{64}",
        "weights": r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf",
    }
    for key, pattern in patterns.items():
        if not re.fullmatch(pattern, checkpoint.get(key, "")):
            raise ValueError(f"Invalid {key} in {manifest}")
    return checkpoint


def pilot_block_capacity(seconds_per_task):
    if not math.isfinite(seconds_per_task) or seconds_per_task <= 0:
        raise ValueError("Pilot duration must be positive and finite")
    return max(1, int(0.90 * HOURS * 3600 / (seconds_per_task * TASKS_PER_BLOCK)))


def post(endpoint, path, payload):
    request = urllib.request.Request(
        endpoint + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(2 ** attempt)


def mode(enabled):
    return {"chat_template_kwargs": {"enable_thinking": enabled}}


def check_template(endpoint):
    messages = drafter_prompt(generate_task(900001, "benign"))
    short = post(endpoint, "/apply-template", {"messages": messages, **mode(False)})["prompt"]
    long = post(endpoint, "/apply-template", {"messages": messages, **mode(True)})["prompt"]
    if short == long or not short.rstrip().endswith("</think>"):
        raise ValueError("Could not verify distinct thinking and non-thinking templates")


def request_completion(endpoint, messages, seed, cap, thinking, model=MODEL, *, template_kwargs=None, grammar=None):
    began = time.monotonic()
    template = mode(thinking) if template_kwargs is None else {"chat_template_kwargs": template_kwargs}
    payload = {
        "model": model, "messages": messages, "seed": seed, "max_tokens": cap,
        "temperature": 0.7, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
        "repeat_penalty": 1.0, **template,
    }
    if grammar is not None:
        payload["grammar"] = grammar
    result = post(endpoint, "/v1/chat/completions", payload)
    choice = result["choices"][0]
    message = choice["message"]
    raw = message.get("content") or ""
    thought = message.get("reasoning_content") or ""
    if "<think>" in raw:
        _, end, final = raw.rpartition("</think>")
        raw = final if end else ""
    usage = result["usage"]
    if usage["prompt_tokens"] <= 0 or not 0 <= usage["completion_tokens"] <= cap:
        raise ValueError("Invalid model-reported token counts")
    return {
        "seed": seed, "max_tokens": cap, "thinking": thinking, "raw": raw,
        "finish_reason": choice["finish_reason"],
        "reasoning_characters": len(thought), "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"], "latency_s": time.monotonic() - began,
    }


def complete(endpoint, messages, seed, cap, thinking, model=MODEL, *, template_kwargs=None):
    record = request_completion(endpoint, messages, seed, cap, thinking, model, template_kwargs=template_kwargs)
    parsed = parse_decision(record["raw"])
    decision = parsed or INVALID
    return {
        **record,
        "decision": {"seller": decision.seller, "quantity": decision.quantity, "shared_fields": list(decision.shared_fields)},
        "parse_ok": parsed is not None,
    }


def collect_effort(endpoint, task, request_seed, model=MODEL):
    messages = drafter_prompt(task)
    short = [
        complete(endpoint, messages, request_seed + i, 256, False, model)
        for i in range(SHORT_DRAFTS)
    ]
    drafts = [Decision.from_json(record["decision"]) for record in short[:2]]
    pick = drafts[select(task, drafts, "opaque")]
    check = evaluate(task, pick, "opaque")
    report = {"seller": pick.seller, **{key: check[key] for key in ("passes", "failed", "visible")}}
    review = complete(endpoint, reviewer_prompt(task, pick, "report", report), request_seed + 8, 256, False, model)
    thinking = {
        str(cap): complete(endpoint, messages, request_seed + i, cap, True, model)
        for i, cap in enumerate(THINK_CAPS, 9)
    }
    return {"short": short, "thinking": thinking, "review": review}


def run_task(endpoint, scenario, model=MODEL):
    market, seed, split = scenario
    task = generate_task(seed, market)
    request_seed = seed * 100 + 25 * MARKETS.index(market)
    return {
        "market": market, "seed": seed, "split": split,
        **collect_effort(endpoint, task, request_seed, model),
    }


def cases_for_block(block):
    return [
        (market, 100001 + 10 * block + offset, "cal" if offset % 5 == 0 else "test")
        for offset in range(10) for market in MARKETS
    ]


def collect_cases(endpoint, cases, model=MODEL):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(lambda scenario: run_task(endpoint, scenario, model), cases))


def run_worker(args):
    checkpoint = load_checkpoint(args.root)
    model = checkpoint["model"]
    worker_dir = args.root / "workers" / args.worker
    worker_dir.mkdir(parents=True, exist_ok=True)
    check_template(args.endpoint)
    pilot_path = worker_dir / "pilot.json"
    if not pilot_path.exists():
        print("Measuring throughput on three pilot tasks", flush=True)
        started = time.monotonic()
        pilot = collect_cases(args.endpoint, [(market, 900100, "pilot") for market in MARKETS], model)
        seconds = time.monotonic() - started
        if any(not any(record["parse_ok"] for record in row["short"]) for row in pilot):
            raise ValueError("All eight short proposals failed parsing on a pilot task")
        write_json(pilot_path, {
            "worker": args.worker, "gpu": args.gpu, "elapsed_s": seconds,
            "seconds_per_task": seconds / len(pilot), "tasks": pilot,
            "model_sha256": checkpoint["model_sha256"],
        })
        print(f"Pilot complete in {seconds:.1f}s", flush=True)

    plan_path = args.root / "plan.json"
    while not plan_path.exists():
        if time.time() >= args.wait_until:
            raise TimeoutError("Timed out waiting for the common study plan")
        time.sleep(5)
    plan = json.loads(plan_path.read_text())
    spec = plan["workers"][args.worker]
    if spec["gpu"] != args.gpu or any(plan[key] != checkpoint[key] for key in CHECKPOINT_KEYS):
        raise ValueError("Worker GPU or checkpoint differs from the plan")
    delay = plan["start_unix"] - time.time()
    if delay > 0:
        time.sleep(delay)
    remaining = plan["deadline_unix"] - time.time()
    if remaining <= 0:
        raise TimeoutError("The collection window has ended")
    signal.alarm(math.ceil(remaining))
    results = args.root / "results" / args.worker
    results.mkdir(parents=True, exist_ok=True)
    done = 0
    for block in range(spec["first_block"], spec["first_block"] + spec["blocks"]):
        path = results / f"block-{block:06d}.json"
        if not path.exists():
            tasks = collect_cases(args.endpoint, cases_for_block(block), model)
            write_json(path, {"worker": args.worker, "block": block, "tasks": tasks})
        done += 1
        print(f"{args.worker}: {done}/{spec['blocks']} blocks collected", flush=True)
    signal.alarm(0)
    write_json(worker_dir / "done.json", {"blocks": done, "tasks": done * TASKS_PER_BLOCK})


def write_plan(args):
    plan_path = args.root / "plan.json"
    if plan_path.exists():
        raise ValueError("The common study plan already exists")
    checkpoint = load_checkpoint(args.root)
    pilots, capacities = {}, {}
    for worker in sorted(set(args.workers)):
        pilot = json.loads((args.root / "workers" / worker / "pilot.json").read_text())
        if pilot.get("model_sha256", MODEL_SHA256) != checkpoint["model_sha256"]:
            raise ValueError(f"Pilot checkpoint differs for {worker}")
        pilots[worker] = pilot
        capacities[worker] = pilot_block_capacity(pilot["seconds_per_task"])
    available = sum(capacities.values())
    total = available if args.blocks is None else args.blocks
    if not len(pilots) <= total <= available:
        raise ValueError(f"{total} blocks cannot be split among {len(pilots)} workers with {available} slots")
    # Give every worker a block, then divide the rest by its remaining pilot capacity.
    specs, offset, cumulative = {}, 0, 0
    for position, (worker, pilot) in enumerate(pilots.items(), 1):
        cumulative += capacities[worker] - 1
        end = position + (total - len(pilots)) * cumulative // max(1, available - len(pilots))
        specs[worker] = {
            "first_block": offset, "blocks": end - offset, "gpu": pilot["gpu"],
            "pilot_seconds_per_task": pilot["seconds_per_task"],
        }
        offset = end
    start = int(time.time()) + 20
    write_json(plan_path, {
        "schema": "decoymarket.distributed_effort.v1",
        **{key: checkpoint[key] for key in CHECKPOINT_KEYS},
        "engine_revision": ENGINE_REVISION, "start_unix": start, "deadline_unix": start + HOURS * 3600,
        "workers": specs, "tasks_per_block": TASKS_PER_BLOCK, "planned_tasks": offset * TASKS_PER_BLOCK,
        "short_drafts": SHORT_DRAFTS, "think_caps": list(THINK_CAPS), "markets": list(MARKETS),
    })
    print(f"Assigned {offset * TASKS_PER_BLOCK} tasks in {offset} blocks across {len(specs)} GPUs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    worker = stages.add_parser("worker")
    worker.add_argument("root", type=Path)
    worker.add_argument("--worker", required=True)
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--endpoint", required=True)
    worker.add_argument("--wait-until", type=int, required=True)
    plan = stages.add_parser("plan")
    plan.add_argument("root", type=Path)
    plan.add_argument("workers", nargs="+")
    plan.add_argument("--blocks", type=int, help="Use this many blocks from the original seed panel")
    args = parser.parse_args()
    if args.stage == "worker":
        run_worker(args)
    else:
        write_plan(args)


if __name__ == "__main__":
    main()
