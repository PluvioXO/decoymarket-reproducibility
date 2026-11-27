"""Collect drafts and reviews for a (model, market) cell and write an episode log.

Drafting does not depend on the disclosure regime (agents see identical text), so each task is
drafted once into M drafts. Each distinct draft decision is then reviewed under three
information conditions: ``none`` (no check result), ``opaque`` (the opaque check report), and
``disclosed`` (the disclosed check report). Review calls are deduplicated by decision because a
review depends only on the selected decision and the condition. Token counts are recorded per
call; nothing here chooses an allocation or looks at hidden outcomes.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .backends import LiveBackend, MockPolicy, parse_decision
from .benchmark import Decision, evaluate, generate_task, optimum
from .prompts import drafter_prompt, reviewer_prompt

INFO_CONDITIONS = ("none", "opaque", "disclosed")
INVALID = Decision(seller="", quantity=-1, shared_fields=())


def decision_key(d: Decision) -> str:
    if not d or not d.seller:
        return "INVALID"
    return json.dumps([d.seller, d.quantity, sorted(d.shared_fields)], sort_keys=True)


@dataclass
class RunConfig:
    model_name: str
    market: str
    seeds: list
    backend: str = "mock"           # mock | live
    n_drafts: int = 16
    max_tokens: int = 1024
    endpoint: str = ""
    model_id: str = ""
    run_seed: int = 0
    token_budget: int = 0           # 0 disables the cap
    mock_rates: dict = None
    draft_only: bool = False


def _mock_tokens(rng, text: str, floor: int) -> dict:
    return {"prompt_tokens": max(floor, len(text) // 4),
            "completion_tokens": rng.randint(40, 130), "latency_s": 0.0}


def _draft_one(cfg, backend, policy, task, rng, idx):
    if cfg.backend == "mock":
        d = policy.draft(task, rng)
        prompt = drafter_prompt(task)
        tok = _mock_tokens(rng, prompt[1]["content"], 200)
        return {"decision": d, "raw": f"[mock draft {idx}] " + decision_key(d),
                "parse_ok": True, **tok}
    seed = cfg.run_seed * 1000 + idx
    out = backend.complete(drafter_prompt(task), cfg.max_tokens, seed=seed)
    d = parse_decision(out["text"])
    return {"decision": d or INVALID, "raw": out["text"], "parse_ok": d is not None, "seed": seed,
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "latency_s": out.get("latency_s", 0.0)}


def _review_one(cfg, backend, policy, task, selected, info, rng, idx):
    report = None
    if info in ("opaque", "disclosed"):
        ev = evaluate(task, selected, info)
        report = {"seller": selected.seller, "passes": ev["passes"],
                  "failed": ev["failed"], "visible": ev["visible"]}
    info_flag = "none" if info == "none" else "report"
    if cfg.backend == "mock":
        rd = policy.review(task, selected, info_flag, report, rng)
        prompt = reviewer_prompt(task, selected, info_flag, report)
        tok = _mock_tokens(rng, prompt[1]["content"], 260)
        return {"decision": rd, "raw": f"[mock review {info}] " + decision_key(rd),
                "parse_ok": True, **tok}
    seed = cfg.run_seed * 1000 + 500 + idx
    out = backend.complete(reviewer_prompt(task, selected, info_flag, report), cfg.max_tokens, seed=seed)
    rd = parse_decision(out["text"])
    return {"decision": rd or selected, "raw": out["text"], "parse_ok": rd is not None, "seed": seed,
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "latency_s": out.get("latency_s", 0.0)}


def run_cell(cfg: RunConfig) -> dict:
    rng = random.Random((cfg.model_name, cfg.market, cfg.run_seed).__hash__())
    backend = None
    policy = None
    if cfg.backend == "live":
        backend = LiveBackend(cfg.endpoint, cfg.model_id)
    else:
        policy = MockPolicy(rates=cfg.mock_rates)

    tasks_out = []
    total_tokens = 0
    stopped = False
    for seed in cfg.seeds:
        if stopped:
            break
        task = generate_task(seed, cfg.market)
        drafts = []
        for i in range(cfg.n_drafts):
            rec = _draft_one(cfg, backend, policy, task, rng, i)
            total_tokens += rec["prompt_tokens"] + rec["completion_tokens"]
            drafts.append(rec)
            if cfg.token_budget and total_tokens > cfg.token_budget:
                stopped = True
                break
        # distinct decisions to review (dedup across identical drafts)
        distinct = {}
        for rec in drafts:
            distinct.setdefault(decision_key(rec["decision"]), rec["decision"])
        reviews = {} if cfg.draft_only else {info: {} for info in INFO_CONDITIONS}
        ridx = 0
        for info in reviews:
            for key, dec in distinct.items():
                if stopped:
                    break
                rr = _review_one(cfg, backend, policy, task, dec, info, rng, ridx)
                ridx += 1
                total_tokens += rr["prompt_tokens"] + rr["completion_tokens"]
                reviews[info][key] = _result_json(rr)
                if cfg.token_budget and total_tokens > cfg.token_budget:
                    stopped = True
        tasks_out.append({
            "task": task.to_json(),
            "optimum": optimum(task),
            "drafts": [{
                **_result_json(r),
                "key": decision_key(r["decision"]),
            } for r in drafts],
            "reviews": reviews,
        })

    return {
        "schema": "decoymarket.episode.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": cfg.backend,
        "is_mock": cfg.backend == "mock",
        "model_name": cfg.model_name,
        "market": cfg.market,
        "n_drafts": cfg.n_drafts,
        "run_seed": cfg.run_seed,
        "endpoint": cfg.endpoint if cfg.backend == "live" else None,
        "model_id": cfg.model_id if cfg.backend == "live" else None,
        "total_tokens": total_tokens,
        "stopped_early": stopped,
        "tasks": tasks_out,
    }


def _dec_json(d: Decision) -> dict:
    return {"seller": d.seller, "quantity": d.quantity, "shared_fields": list(d.shared_fields)}


def _result_json(result: dict) -> dict:
    return {**result, "decision": _dec_json(result["decision"])}


def write_episode(cfg: RunConfig, out_path: str) -> dict:
    episode = run_cell(cfg)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"refusing to overwrite existing episode {p}")
    p.write_text(json.dumps(episode, indent=2) + "\n")
    return episode
