"""Small pure-Python numerics: digamma, Dirichlet-multinomial fit, logistic fit, bootstrap.

No third-party dependency, so the analysis runs in any Python 3.10+.
"""
from __future__ import annotations

import math
import random


def digamma(x: float) -> float:
    """Psi function via recurrence up to x>=6 then an asymptotic expansion."""
    result = 0.0
    while x < 6:
        result -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    result += math.log(x) - 1.0 / (2.0 * x) - f * (
        1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0))))
    return result


def fit_dirichlet_multinomial(count_rows, iters: int = 400, tol: float = 1e-9):
    """Minka fixed-point MLE of a Dirichlet from multinomial count rows (Polya model)."""
    k = len(count_rows[0])
    totals = [sum(r) for r in count_rows]
    # method-of-moments initialization
    props = [[(r[j] + 0.5) / (t + 0.5 * k) for j in range(k)] for r, t in zip(count_rows, totals)]
    mean = [sum(p[j] for p in props) / len(props) for j in range(k)]
    m2 = [sum(p[j] ** 2 for p in props) / len(props) for j in range(k)]
    denom = 0.0
    for j in range(k):
        var = max(m2[j] - mean[j] ** 2, 1e-6)
        denom += math.log((mean[j] * (1 - mean[j]) / var) - 1 + 1e-9) if 0 < mean[j] < 1 else 0.0
    s0 = math.exp(denom / k) if denom else float(k)
    alpha = [max(mean[j] * s0, 1e-3) for j in range(k)]
    for _ in range(iters):
        a0 = sum(alpha)
        num = [0.0] * k
        den = 0.0
        for r, t in zip(count_rows, totals):
            den += digamma(t + a0) - digamma(a0)
            for j in range(k):
                num[j] += digamma(r[j] + alpha[j]) - digamma(alpha[j])
        if den <= 0:
            break
        new = [max(alpha[j] * num[j] / den, 1e-6) for j in range(k)]
        if max(abs(new[j] - alpha[j]) for j in range(k)) < tol:
            alpha = new
            break
        alpha = new
    return alpha


def beta_power_moment(a_plus: float, a0: float, k: int) -> float:
    """E[p^k] where p ~ Beta(a_plus, a0 - a_plus): prod_{i=0}^{k-1} (a_plus+i)/(a0+i)."""
    out = 1.0
    for i in range(k):
        out *= (a_plus + i) / (a0 + i)
    return out


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(X, y, l2: float = 1e-3, lr: float = 0.3, iters: int = 20000):
    """Minimize summed squared error of a sigmoid link by gradient descent (small problems)."""
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    for _ in range(iters):
        grad = [0.0] * d
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(d))
            p = sigmoid(z)
            err = (p - yi) * p * (1 - p)
            for j in range(d):
                grad[j] += err * xi[j]
        for j in range(d):
            grad[j] = grad[j] / n + l2 * w[j]
        for j in range(d):
            w[j] -= lr * grad[j]
    return w


def predict_logistic(w, x) -> float:
    return sigmoid(sum(w[j] * x[j] for j in range(len(w))))


def bootstrap_ci(values, statistic, draws: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile interval of a statistic over task-level resamples of a list of per-task values."""
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * draws)]
    hi = stats[int((1 - alpha / 2) * draws) - 1]
    return statistic(values), lo, hi
