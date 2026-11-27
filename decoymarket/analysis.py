"""Load episodes, build exact frontiers, fit the rates forecaster and baselines, test hypotheses.

Everything downstream of the model calls lives here. The analysis is deterministic given the
episode logs. Budgets are expressed in draft-equivalents (mean draft token cost in the cell).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import estimators as es
from . import mathutil as mu
from .benchmark import (
    Decision, _offer_by_seller, classify_failure, evaluate, optimum, task_from_json,
)
from .run import INFO_CONDITIONS

KS = list(range(1, 17))


@dataclass
class TaskData:
    task: object
    drafts: list           # list[Decision]
    dtok: list             # tokens per draft
    reviews: dict          # info -> list[Decision] aligned to drafts
    rtok: dict             # info -> list[int] aligned to drafts
    optimum: dict


def _dec(d: dict) -> Decision:
    return Decision(d["seller"], d.get("quantity", -1), tuple(d.get("shared_fields", [])))


def load_episode(path):
    data = json.loads(Path(path).read_text())
    tds = []
    for row in data["tasks"]:
        task = task_from_json(row["task"])
        drafts = [_dec(d["decision"]) for d in row["drafts"]]
        dtok = [d["prompt_tokens"] + d["completion_tokens"] for d in row["drafts"]]
        keys = [d["key"] for d in row["drafts"]]
        reviews, rtok = {}, {}
        for info in INFO_CONDITIONS:
            table = row["reviews"][info]
            reviews[info] = [_dec(table[k]["decision"]) for k in keys]
            rtok[info] = [table[k]["prompt_tokens"] + table[k]["completion_tokens"] for k in keys]
        tds.append(TaskData(task, drafts, dtok, reviews, rtok, row["optimum"]))
    return data, tds


def classify(task, decision, regime: str) -> str:
    if classify_failure(task, decision)["success"]:
        return "S"
    ev = evaluate(task, decision, regime)
    opt_off = _offer_by_seller(task, optimum(task)["seller"])
    opt_vis = opt_off.visible_total(task.mandate.quantity)
    if ev["passes"] and ev["visible"] < opt_vis:
        return "A"
    if ev["passes"]:
        return "B"
    return "N"


def _review_key(regime: str, report_info: str) -> str:
    return "none" if report_info == "none" else regime


def mean_draft_cost(tds) -> float:
    tot = sum(sum(td.dtok) for td in tds)
    n = sum(len(td.dtok) for td in tds)
    return tot / n if n else 1.0


# --------------------------------------------------------------------------- frontiers

def frontier(tds, regime: str, report_info: str):
    """Per-task success and cost arrays for no-review and one-review allocations, plus AGREE2."""
    rk = _review_key(regime, report_info)
    unit = mean_draft_cost(tds)
    out = {"k": KS, "no_review": {k: [] for k in KS}, "review": {k: [] for k in KS},
           "cost0": {k: [] for k in KS}, "cost1": {k: [] for k in KS}, "agree2": []}
    for td in tds:
        c0 = es.no_review_curve(td.task, td.drafts, td.dtok, regime, KS)
        c1 = es.review_curve(td.task, td.drafts, td.dtok, td.reviews[rk], td.rtok[rk], regime, KS)
        for k in KS:
            out["no_review"][k].append(c0[k]["success"])
            out["cost0"][k].append(c0[k]["cost"] / unit)
            out["review"][k].append(c1[k]["success"])
            out["cost1"][k].append(c1[k]["cost"] / unit)
        a2 = es.agree2(td.task, td.drafts, td.dtok, td.reviews[rk], td.rtok[rk], regime)
        out["agree2"].append((a2["success"], a2["cost"] / unit))
    out["unit_tokens"] = unit
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summary_frontier(fr):
    s = {"k": KS, "unit_tokens": fr["unit_tokens"]}
    s["no_review"] = {k: _mean(fr["no_review"][k]) for k in KS}
    s["review"] = {k: _mean(fr["review"][k]) for k in KS}
    s["cost0"] = {k: _mean(fr["cost0"][k]) for k in KS}
    s["cost1"] = {k: _mean(fr["cost1"][k]) for k in KS}
    s["agree2"] = (_mean([a for a, _ in fr["agree2"]]), _mean([c for _, c in fr["agree2"]]))
    return s


# --------------------------------------------------------------------------- rates model

def fit_rates(calib_tds, regime: str, report_info: str, m: int = 4):
    rk = _review_key(regime, report_info)
    rows = []
    rev = {"S": [0, 0], "A": [0, 0], "B": [0, 0], "N": [0, 0]}  # [successes, count]
    for td in calib_tds:
        c = {"S": 0, "A": 0, "B": 0, "N": 0}
        for i in range(min(m, len(td.drafts))):
            cat = classify(td.task, td.drafts[i], regime)
            c[cat] += 1
            ok = 1 if classify_failure(td.task, td.reviews[rk][i])["success"] else 0
            rev[cat][0] += ok
            rev[cat][1] += 1
        rows.append([c["S"], c["A"], c["B"], c["N"]])
    alpha = mu.fit_dirichlet_multinomial(rows)
    aS, aA, aB, aN = alpha
    a0 = sum(alpha)
    overall = (sum(r[0] for r in rev.values()) / max(sum(r[1] for r in rev.values()), 1))

    def rate(cat):
        s, n = rev[cat]
        return s / n if n else overall

    phiA, phiB, phiN = rate("A"), rate("B"), rate("N")
    beta = 1 - rate("S")

    def predict(k, r):
        E1 = mu.beta_power_moment(a0 - aA, a0, k)          # E[(1-pA)^k]
        E2 = mu.beta_power_moment(a0 - aA - aS, a0, k)     # E[(1-pA-pS)^k]
        EN = mu.beta_power_moment(aN, a0, k)               # E[pN^k]
        p0 = E1 - E2
        if r == 0:
            return max(0.0, min(1.0, p0))
        p1 = (1 - beta) * (E1 - E2) + phiA * (1 - E1) + phiB * (E2 - EN) + phiN * EN
        return max(0.0, min(1.0, p1))

    return {"alpha": alpha, "beta": beta, "phiA": phiA, "phiB": phiB, "phiN": phiN,
            "predict": predict}


def fit_logistic_baseline(calib_summary):
    X, y = [], []
    for k in (1, 2, 3, 4):
        for r in (0, 1):
            X.append([1.0, math.log(k), float(r), math.log(k) * r])
            y.append(calib_summary["review"][k] if r else calib_summary["no_review"][k])
    w = mu.fit_logistic(X, y)
    return lambda k, r: mu.predict_logistic(w, [1.0, math.log(k), float(r), math.log(k) * r])


def review_cost_ratio(tds, regime, report_info):
    rk = _review_key(regime, report_info)
    unit = mean_draft_cost(tds)
    rtot = sum(sum(td.rtok[rk]) for td in tds)
    rn = sum(len(td.rtok[rk]) for td in tds)
    return (rtot / rn) / unit if rn else 1.0


def alloc_cost(k, r, ratio):
    return k + (ratio if r else 0.0)


def best_alloc_at_budget(predict, budget, ratio, ks=KS):
    best = None
    for k in ks:
        for r in (0, 1):
            if alloc_cost(k, r, ratio) <= budget + 1e-9:
                p = predict(k, r)
                if best is None or p > best[0]:
                    best = (p, k, r)
    return best  # (predicted_success, k, r)


# --------------------------------------------------------------------------- hypotheses

def paired_delta_ci(per_task_a, per_task_b, seed=0):
    diffs = [a - b for a, b in zip(per_task_a, per_task_b)]
    return mu.bootstrap_ci(diffs, _mean, draws=10000, seed=seed)


def run_analysis(calib_episode, test_episode, regime="opaque", report_info="report"):
    _, calib = load_episode(calib_episode)
    meta, test = load_episode(test_episode)
    market = meta["market"]
    res = {"market": market, "regime": regime, "report_info": report_info, "is_mock": meta["is_mock"]}

    fr = frontier(test, regime, report_info)
    sm = summary_frontier(fr)
    res["frontier"] = sm
    res["frontier_disclosed"] = summary_frontier(frontier(test, "disclosed", report_info))

    # khat from calibration (argmax over k<=4 of no-review success)
    cfr = summary_frontier(frontier(calib, regime, report_info))
    khat = max((1, 2, 3, 4), key=lambda k: cfr["no_review"][k])
    res["khat"] = khat

    # H1: decoy P(16,0) < P(khat,0); benign P(16,0) > P(4,0)
    d16 = fr["no_review"][16]
    dkhat = fr["no_review"][khat]
    d4 = fr["no_review"][4]
    res["p1"] = sm["no_review"][1]
    res["p4"] = sm["no_review"][4]
    res["pkhat"] = sm["no_review"][khat]
    res["p16"] = sm["no_review"][16]
    res["p16_disclosed"] = res["frontier_disclosed"]["no_review"][16]
    res["peak_k"] = max(KS, key=lambda k: sm["no_review"][k])
    res["peak"] = sm["no_review"][res["peak_k"]]
    if market == "benign":
        res["H1_delta"], res["H1_lo"], res["H1_hi"] = paired_delta_ci(d16, d4, seed=1)
    else:
        res["H1_delta"], res["H1_lo"], res["H1_hi"] = paired_delta_ci(dkhat, d16, seed=1)

    # H2: disclosed P(16,0) > opaque P(16,0)
    fr_disc = frontier(test, "disclosed", report_info)
    res["H2_delta"], res["H2_lo"], res["H2_hi"] = paired_delta_ci(
        fr_disc["no_review"][16], fr["no_review"][16], seed=2)

    # H3: at budget 8 draft-equivalents, best r=1 vs best r=0 (per task)
    ratio = review_cost_ratio(test, regime, report_info)
    res["review_cost_ratio"] = ratio
    kbud = 8
    # best fixed k within budget for r=0 and for r=1 (chosen on test means, reported descriptively)
    k0 = max([k for k in KS if alloc_cost(k, 0, ratio) <= kbud], key=lambda k: sm["no_review"][k])
    k1 = max([k for k in KS if alloc_cost(k, 1, ratio) <= kbud], key=lambda k: sm["review"][k])
    res["H3_k0"], res["H3_k1"] = k0, k1
    res["H3_delta"], res["H3_lo"], res["H3_hi"] = paired_delta_ci(
        fr["review"][k1], fr["no_review"][k0], seed=3)

    # H5: AGREE2 vs best fixed allocation costing <= AGREE2 cost (decoy)
    a2_succ = _mean([a for a, _ in fr["agree2"]])
    a2_cost = _mean([c for _, c in fr["agree2"]])
    res["agree2_success"], res["agree2_cost"] = a2_succ, a2_cost
    cand = []
    for k in KS:
        if alloc_cost(k, 0, ratio) <= a2_cost + 1e-9:
            cand.append(("0", k, sm["no_review"][k]))
        if alloc_cost(k, 1, ratio) <= a2_cost + 1e-9:
            cand.append(("1", k, sm["review"][k]))
    best_fixed = max(cand, key=lambda c: c[2]) if cand else ("0", 1, sm["no_review"][1])
    res["H5_best_fixed"] = {"r": best_fixed[0], "k": best_fixed[1], "success": best_fixed[2]}
    # per-task AGREE2 minus the chosen best-fixed allocation's per-task success
    per_task_fixed = (fr["review"] if best_fixed[0] == "1" else fr["no_review"])[best_fixed[1]]
    per_task_a2 = [a for a, _ in fr["agree2"]]
    res["H5_delta"], res["H5_lo"], res["H5_hi"] = paired_delta_ci(per_task_a2, per_task_fixed, seed=5)

    # H6: A-draft repair rate under report vs none (decoy)
    rep_none = _a_repair(test, regime, "none")
    rep_report = _a_repair(test, regime, "report")
    res["H6_repair_none"] = rep_none
    res["H6_repair_report"] = rep_report
    res["H6_delta"] = (rep_report - rep_none) if (rep_none is not None and rep_report is not None) else None

    # H4: forecasting
    res["forecast"] = forecasting(calib, test, regime, report_info, ratio)
    return res


def _a_repair(tds, regime, report_info):
    rk = _review_key(regime, report_info)
    succ = n = 0
    for td in tds:
        for i, d in enumerate(td.drafts):
            if classify(td.task, d, regime) == "A":
                n += 1
                if classify_failure(td.task, td.reviews[rk][i])["success"]:
                    succ += 1
    return succ / n if n else None


def forecasting(calib, test, regime, report_info, ratio):
    rates = fit_rates(calib, regime, report_info)
    csum = summary_frontier(frontier(calib, regime, report_info))
    logit = fit_logistic_baseline(csum)
    tsum = summary_frontier(frontier(test, regime, report_info))

    def mae(pred):
        errs = []
        for k in KS:
            errs.append(abs(pred(k, 0) - tsum["no_review"][k]))
            errs.append(abs(pred(k, 1) - tsum["review"][k]))
        return _mean(errs)

    def actual(k, r):
        return tsum["review"][k] if r else tsum["no_review"][k]

    def regret(pred, budget):
        _, k, r = best_alloc_at_budget(pred, budget, ratio)
        # best actual at budget
        best = max((actual(kk, rr) for kk in KS for rr in (0, 1)
                    if alloc_cost(kk, rr, ratio) <= budget + 1e-9))
        return best - actual(k, r)

    # scale-up baseline: pick best calib allocation at budget 4, scale k to the target budget
    base = best_alloc_at_budget(lambda k, r: csum["review"][k] if r else csum["no_review"][k],
                                4, ratio)
    _, bk, br = base

    def scaleup(k, r):  # as a predictor it just mirrors actual for its own chosen alloc
        return actual(k, r)

    def scaleup_regret(budget):
        kk = max([k for k in KS if alloc_cost(k, br, ratio) <= budget + 1e-9], default=bk)
        best = max((actual(a, b) for a in KS for b in (0, 1)
                    if alloc_cost(a, b, ratio) <= budget + 1e-9))
        return best - actual(kk, br)

    return {
        "rates_mae": mae(rates["predict"]),
        "logit_mae": mae(logit),
        "rates_regret8": regret(rates["predict"], 8),
        "rates_regret16": regret(rates["predict"], 16),
        "logit_regret8": regret(logit, 8),
        "logit_regret16": regret(logit, 16),
        "scaleup_regret8": scaleup_regret(8),
        "scaleup_regret16": scaleup_regret(16),
        "rates_params": {k: rates[k] for k in ("beta", "phiA", "phiB", "phiN")},
        "rates_pick8": best_alloc_at_budget(rates["predict"], 8, ratio),
        "rates_pick16": best_alloc_at_budget(rates["predict"], 16, ratio),
    }
