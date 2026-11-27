"""Model backends and the decision parser.

LiveBackend talks to any OpenAI-compatible /v1/chat/completions endpoint (vLLM or llama.cpp
--api). MockPolicy is an explicit software stub: it fabricates decisions with tunable category
rates so the pipeline can be tested without a model. Mock output is clearly labelled and must
never be read as evidence about a language model.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .benchmark import (
    Decision, Task, canonical_decision, classify_failure, evaluate, feasible_offers,
    optimum, _offer_by_seller,
)

# --------------------------------------------------------------------------- parsing

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_decision(text: str) -> Decision | None:
    """Extract the last JSON object with a 'seller' field from model text."""
    if not text:
        return None
    candidates = _JSON_OBJ.findall(text)
    for chunk in reversed(candidates):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "seller" in obj:
            try:
                return Decision.from_json(obj)
            except (KeyError, ValueError, TypeError):
                continue
    return None


# --------------------------------------------------------------------------- live backend

class LiveBackend:
    def __init__(self, endpoint: str, model: str, temperature: float = 0.7,
                 top_p: float = 0.95, timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.api_key = os.environ.get("SWARM_API_KEY", "")

    def complete(self, messages, max_tokens: int, seed: int) -> dict:
        url = self.endpoint
        if not url.endswith("/chat/completions"):
            url = url.removesuffix("/v1") + "/v1/chat/completions"
        payload = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "top_p": self.top_p, "max_tokens": max_tokens, "seed": seed,
        }
        if self.model == "Qwen/Qwen3-8B":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        usage = body.get("usage", {})
        return {
            "text": body["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": time.time() - t0,
        }


# --------------------------------------------------------------------------- mock policy

class MockPolicy:
    """Task-aware fabrication for software tests. NOT a model.

    Drafter: with the configured probabilities it returns the optimum (S), a decoy (A, decoy
    markets only), a feasible-but-suboptimal offer (B), or an opaque-failing offer (N). Reviewer:
    if shown a report that flags the decision, or on its own with probability ``repair``, it
    returns the optimum; otherwise it keeps the selection but may break a good one with
    probability ``brk``.
    """

    def __init__(self, rates=None, repair: float = 0.6, brk: float = 0.05, report_boost: float = 0.9):
        self.rates = rates or {"S": 0.45, "A": 0.35, "B": 0.1, "N": 0.1}
        self.repair = repair
        self.brk = brk
        self.report_boost = report_boost

    def _buckets(self, task: Task):
        m = task.mandate
        opt = optimum(task)["seller"]
        decoys = list(task.decoy_sellers)
        feas_b = [o.seller for o in feasible_offers(task) if o.seller != opt]
        fails = [o.seller for o in task.offers
                 if not evaluate(task, canonical_decision(o, m), "opaque")["passes"]]
        return opt, decoys, feas_b, fails

    def draft(self, task: Task, rng) -> Decision:
        m = task.mandate
        opt, decoys, feas_b, fails = self._buckets(task)
        cats, weights = zip(*self.rates.items())
        cat = rng.choices(cats, weights=weights, k=1)[0]
        if cat == "A" and not decoys:
            cat = "B" if feas_b else "S"
        if cat == "B" and not feas_b:
            cat = "S"
        if cat == "N" and not fails:
            cat = "S"
        if cat == "S":
            seller = opt
        elif cat == "A":
            seller = rng.choice(decoys)
        elif cat == "B":
            seller = rng.choice(feas_b)
        else:
            seller = rng.choice(fails)
        offer = _offer_by_seller(task, seller)
        shared = tuple(sorted(offer.required_fields))
        return Decision(seller=seller, quantity=m.quantity, shared_fields=shared)

    def review(self, task: Task, selected: Decision, info: str, report: dict, rng) -> Decision:
        m = task.mandate
        opt = optimum(task)
        opt_dec = Decision(opt["seller"], opt["quantity"], tuple(opt["shared_fields"]))
        flagged = info == "report" and report is not None and not report["passes"]
        sel_ok = classify_failure(task, selected)["success"]
        p = self.repair
        if flagged:
            p = self.report_boost
        if not sel_ok and rng.random() < p:
            return opt_dec
        if sel_ok and rng.random() < self.brk:
            # break a good selection by switching to a feasible-suboptimal offer if any
            _, _, feas_b, _ = self._buckets(task)
            if feas_b:
                o = _offer_by_seller(task, rng.choice(feas_b))
                return Decision(o.seller, m.quantity, tuple(sorted(o.required_fields)))
        return selected
