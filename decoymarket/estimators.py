"""Exact subsampling estimators over a fixed pool of drafts and reviews.

For a task we hold M drafts and, for each draft, one review under each information condition.
Drawing a random k-subset of the M drafts and submitting the selector's pick gives an exact
distribution over outcomes, so every allocation with k<=M is estimated without resampling.

The weight that the selector's pick is draft at sorted position p (best first) in a random
k-subset is C(M-1-p, k-1) / C(M, k); these weights sum to one (hockey-stick identity) and
generalize the pass@k estimator.
"""
from __future__ import annotations

from math import comb

from .benchmark import Decision, Task, classify_failure, selection_key


def _sorted_positions(task: Task, drafts, regime: str):
    """Return draft indices ordered best-first under the regime's selection key."""
    order = sorted(range(len(drafts)),
                   key=lambda i: selection_key(task, drafts[i], regime) + (i,))
    return order


def selection_weights(m: int, order, k: int):
    """weight[i] = P(draft i is the selector's pick) for a uniform k-subset of m drafts."""
    denom = comb(m, k)
    w = [0.0] * m
    for pos, i in enumerate(order):
        w[i] = comb(m - 1 - pos, k - 1) / denom if k - 1 <= m - 1 - pos else 0.0
    return w


def _is_success(task: Task, decision: Decision) -> float:
    return 1.0 if classify_failure(task, decision)["success"] else 0.0


def no_review_curve(task: Task, drafts, tokens, regime: str, ks):
    """Success probability and expected token cost at each k with no review."""
    m = len(drafts)
    succ = [_is_success(task, d) for d in drafts]
    mean_tok = sum(tokens) / m
    out = {}
    for k in ks:
        if k > m:
            continue
        w = selection_weights(m, _sorted_positions(task, drafts, regime), k)
        out[k] = {"success": sum(w[i] * succ[i] for i in range(m)),
                  "cost": k * mean_tok, "reviews": 0}
    return out


def review_curve(task: Task, drafts, tokens, reviews, review_tokens, regime: str, ks):
    """Success and cost at each k with one review of the selector's pick.

    reviews[i] is the reviewed decision when draft i is the selected one, under the information
    condition already chosen by the caller; review_tokens[i] is that review's token count.
    """
    m = len(drafts)
    rsucc = [_is_success(task, rd) for rd in reviews]
    mean_tok = sum(tokens) / m
    out = {}
    for k in ks:
        if k > m:
            continue
        w = selection_weights(m, _sorted_positions(task, drafts, regime), k)
        out[k] = {"success": sum(w[i] * rsucc[i] for i in range(m)),
                  "cost": k * mean_tok + sum(w[i] * review_tokens[i] for i in range(m)),
                  "reviews": 1}
    return out


def agree2(task: Task, drafts, tokens, reviews, review_tokens, regime: str):
    """AGREE2: draw two drafts; submit if they agree and pass, else review the selector's pick."""
    m = len(drafts)
    pairs = 0
    succ = 0.0
    cost = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pairs += 1
            di, dj = drafts[i], drafts[j]
            # selector's pick within the pair
            pick = i if selection_key(task, di, regime) + (i,) <= selection_key(task, dj, regime) + (j,) else j
            agree = (di.seller, di.quantity, tuple(di.shared_fields)) == \
                    (dj.seller, dj.quantity, tuple(dj.shared_fields))
            same_and_pass = agree and _passes_check(task, di, regime)
            if same_and_pass:
                final = di
                c = tokens[i] + tokens[j]
            else:
                final = reviews[pick]
                c = tokens[i] + tokens[j] + review_tokens[pick]
            succ += _is_success(task, final)
            cost += c
    return {"success": succ / pairs, "cost": cost / pairs, "pairs": pairs}


def _passes_check(task: Task, decision: Decision, regime: str) -> bool:
    from .benchmark import evaluate
    return evaluate(task, decision, regime)["passes"]
