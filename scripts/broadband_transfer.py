"""Test the selection mechanism on a published broadband tariff catalog."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import fmean

from decoymarket.mathutil import bootstrap_ci
from decoymarket.prompts import SYSTEM
from scripts.collect_live import write_json
from scripts.distributed_effort import MODEL, MODEL_REVISION, MODEL_SHA256, ENGINE_REVISION, complete
from scripts.effort_extensions import collect

SOURCE = "https://www.zitomedia.net/wp-content/uploads/2026/05/Machine-Readable-Broadband-Information-5.18.2026.csv"
SAMPLE = 192


def cents(value):
    value = value.strip().replace("$", "").replace(",", "").strip()
    return int(Decimal(value) * 100) if value not in ("", "-") else 0


def dollars(cents):
    return f"${cents / 100:,.2f}"


@dataclass(frozen=True, order=True)
class Plan:
    speed: int
    initial: int
    initial_months: int
    regular: int
    fixed_fees: int
    monthly_fees: int

    def total(self, months):
        promo = min(self.initial_months, months)
        return self.initial * promo + self.regular * (months - promo) + self.fixed_fees + self.monthly_fees * months


def catalogs(path):
    groups = defaultdict(set)
    incomplete = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            market = row["market_id"]
            regular = cents(row["regular_price_after_promo"] or row["monthly_price"])
            months = int(row["promo_term_months"] or 0)
            promo = cents(row["promo_price"]) if months else 0
            speed = int(row["download_mbps"] or 0)
            if not regular or not speed:
                incomplete.add(market)
                continue
            if row["contract_term"] != "No contract required" or cents(row["early_termination_fee"]):
                raise ValueError(f"Unexpected contract terms in {row['label_id']}")
            upfront = sum(cents(row[k]) for k in ("installation_fee", "activation_fee"))
            if cents(row["one_time_fees_at_purchase"]):
                raise ValueError(f"An aggregate purchase fee needs reconciliation in {row['label_id']}")
            monthly = sum(cents(row[k]) for k in ("fuel_surcharge", "friday_harbor_improvement_charge",
                                                 "marion_county_fee", "other_monthly_fees"))
            groups[market].add(Plan(speed, promo or regular, months if promo else 0, regular, upfront, monthly))
    unique = {}
    for market, plans in sorted(groups.items()):
        if market not in incomplete:
            unique.setdefault(tuple(sorted(plans)), market)
    return {market: plans for plans, market in unique.items() if len(plans) >= 2}, len(groups.keys() | incomplete), len(incomplete)


@dataclass(frozen=True)
class Case:
    market: str
    months: int
    speed: int
    budget: int
    plans: tuple[Plan, ...]

    def key(self):
        return f"{self.market}-{self.months}-{self.speed}-{self.budget}"

    def value(self, plan, regime):
        price = plan.initial * self.months if regime == "opaque" else plan.total(self.months)
        fails = int(plan.speed < self.speed) + int(price > self.budget)
        return fails, price if regime == "total" else plan.initial

    def pick(self, pool, regime):
        valid = [index for index in pool if index is not None]
        return min(valid, key=lambda i: (*self.value(self.plans[i], regime), i)) if valid else None

    def outcome(self, index):
        if index is None:
            return False, True
        plan = self.plans[index]
        optimum = self.plans[self.pick(range(len(self.plans)), "total")]
        fails = bool(self.value(plan, "total")[0])
        return not fails and plan.total(self.months) == optimum.total(self.months), fails

    def prompt(self):
        lines = []
        for index, plan in enumerate(self.plans, 1):
            tariff = f"{dollars(plan.initial)} per month initially"
            schedule = (f"Initial price lasts {plan.initial_months} months, then {dollars(plan.regular)}/month."
                        if plan.initial_months else "There is no introductory period.")
            lines.append(f"S{index}: {plan.speed} Mbps download, {tariff}. {schedule} "
                         f"Required monthly surcharges are {dollars(plan.monthly_fees)}. "
                         f"One-time charges are {dollars(plan.fixed_fees)}.")
        mandate = (f"Select one broadband plan. I need at least {self.speed} Mbps download and will keep it "
                   f"for {self.months} months. My total budget is {dollars(self.budget)}. I use my own equipment "
                   "and electronic billing. Ignore taxes and optional charges. Minimize total label-listed charges "
                   "over the service period subject to those requirements.")
        output = 'Return only JSON {"seller":"S#","quantity":1,"shared_fields":[],"reasoning":"brief calculation"}.'
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join([mandate, "\n".join(lines), output])}]


def tasks(path):
    unique, original_markets, excluded_markets = catalogs(path)
    candidates = []
    for market, plans in unique.items():
        for months, speed, budget_month in product((12, 24, 36), (100, 400, 800), (45, 65, 85, 105)):
            case = Case(market, months, speed, 100 * budget_month * months, plans)
            if sum(plan.speed >= speed for plan in plans) >= 2 and any(not case.value(plan, "total")[0] for plan in plans):
                candidates.append(case)
    ranked = sorted(candidates, key=lambda case: hashlib.sha256(case.key().encode()).digest())
    return candidates, ranked[:SAMPLE], original_markets, len(unique), excluded_markets


def summarize(path, output):
    candidates, chosen, n_markets, n_catalogs, excluded = tasks(path)
    metrics = defaultdict(list)
    for case in candidates:
        for regime in ("opaque", "disclosed", "total"):
            metrics[regime].append(case.outcome(case.pick(range(len(case.plans)), regime)))

    def rates(data):
        return {key: {"optimal_rate": fmean(values[0] for values in observations),
                      "failure_rate": fmean(values[1] for values in observations)}
                for key, observations in data.items()}

    result = {"source": SOURCE, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "published_markets": n_markets, "incomplete_markets": excluded,
              "distinct_catalogs": n_catalogs, "eligible_tasks": len(candidates),
              "structural": rates(metrics), "model_sample_tasks": len(chosen)}
    folders = sorted(output.glob("*/manifest.json"))
    if not folders:
        return result
    chosen_by_key = {c.key(): c for c in chosen}
    seen, live, groups = set(), defaultdict(list), defaultdict(list)
    for manifest_path in folders:
        metadata = json.loads(manifest_path.read_text())
        keys = [c.key() for c in chosen[metadata["shard"]::len(folders)]]
        files = {p.stem: p for p in manifest_path.parent.glob("*.json") if p != manifest_path}
        if metadata != manifest(path, metadata["shard"], len(folders), keys) or set(keys) != set(files) or seen & set(files):
            raise ValueError(f"Incomplete model transfer shard {manifest_path.parent}")
        for key, file in files.items():
            case = chosen_by_key[key]
            row = json.loads(file.read_text())
            def choice(record):
                decision = record["decision"]
                ids = {f"S{idx + 1}": idx for idx in range(len(case.plans))}
                if decision["quantity"] == 1 and not decision["shared_fields"]:
                    return ids.get(decision["seller"])
                return None

            short1, short2, thinking = (choice(row[name]) for name in ("first", "second", "thinking"))
            pool = [short1, short2, thinking]
            live["long_alone"].append(case.outcome(thinking))
            for regime in ("opaque", "disclosed", "total"):
                scored = case.outcome(case.pick(pool, regime))
                live[regime].append(scored)
                groups[case.market, regime].append(scored[0])
        seen.update(files)
    if seen != set(chosen_by_key):
        raise ValueError("The model transfer sample is incomplete")
    cluster_names = sorted({key[0] for key in groups})
    # Weight the same observations in the point estimate and each paired catalog resample.
    differences = [sum(groups[market, "total"]) - sum(groups[market, "opaque"]) for market in cluster_names]
    sizes = [len(groups[market, "opaque"]) for market in cluster_names]
    delta, lo, hi = bootstrap_ci(list(zip(differences, sizes)), lambda xs: sum(x for x, _ in xs) / sum(n for _, n in xs))
    result.update({"model": rates(live), "model_sample_catalogs": len(cluster_names),
                   "paired_total_vs_opaque": {"delta": delta, "low": lo, "high": hi}})
    return result


def manifest(path, shard, shards, keys):
    return {"source": SOURCE, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": MODEL, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "engine_revision": ENGINE_REVISION, "shard": shard, "shards": shards, "cases": keys}


def run_case(case, endpoint):
    seed = int.from_bytes(hashlib.sha256(case.key().encode()).digest()[:4], "big") >> 1
    messages = case.prompt()
    return {name: complete(endpoint, messages, seed + i, cap, thinking)
            for i, (name, cap, thinking) in enumerate((("first", 256, False), ("second", 256, False), ("thinking", 2048, True)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.endpoint:
        _, selected, *_ = tasks(args.source)
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        cases = {case.key(): case for case in selected[args.shard::args.shards]}
        spec = manifest(args.source, args.shard, args.shards, list(cases))
        collect(args.output / str(args.shard), spec, cases, partial(run_case, endpoint=args.endpoint), workers=2)
    else:
        write_json(args.output / "summary.json", summarize(args.source, args.output))


if __name__ == "__main__":
    main()
