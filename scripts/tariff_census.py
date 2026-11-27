"""Apply the broadband disclosure replay to additional published plan portfolios."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
from itertools import combinations, product
from pathlib import Path

from scripts.broadband_transfer import Case, Plan, cents, tasks
from scripts.collect_live import write_json


SOURCES = {
    "gwi.csv": "https://www.gwi.net/gwi-broadband-labels.csv",
    "mitchell.csv": "https://mitchelltelecom.com/wp-content/uploads/2026/03/BroadbandLabels-RES-2026.csv",
    **{f"nkt-{name}.csv": f"https://nktelco.com/wp-content/uploads/2026/05/NKT-Readable-Format-{name}.csv"
       for name in ("5GIG", "2GIG", "Standard-1GIG", "Basic-500M", "1GIG", "500M")},
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [{key.strip(): "" if value.strip() == "NULL" else value.strip() for key, value in row.items()}
                for row in csv.DictReader(source)]


def plan(row, upfront):
    if row["connection_type"] != "Fixed" or row["contract_req"] != "No":
        raise ValueError(f"Unexpected contract on {row['unique_plan_id']}")
    regular = cents(row["monthly_price"])
    if row["intro_rate"] == "Yes":
        months = int(row["intro_rate_time"].split()[0])
        initial = cents(row["intro_rate_price"])
    elif row["intro_rate"] == "No":
        months, initial = 0, regular
    else:
        raise ValueError(f"Unknown introductory rate on {row['unique_plan_id']}")
    speed = int(Decimal(row["typical_download_speed"].removesuffix(" Mbps")))
    if speed <= 0 or regular <= 0 or initial <= 0 or (row["intro_rate"] == "Yes" and months <= 0):
        raise ValueError(f"Unexpected tariff on {row['unique_plan_id']}")
    return Plan(speed, initial, months, regular, upfront, 0)


def portfolios(folder):
    gwi = [row for row in read_rows(folder / "gwi.csv") if "Residential" in row["service_plan_name"]]
    if any(row["single_purchase_fee_descr"] != "Installation fee" or "(optional)" not in row["monthly_provider_fee_descr"]
           for row in gwi):
        raise ValueError("GWI fee categories changed")

    mitchell = []
    mitchell_all_fees = []
    for row in read_rows(folder / "mitchell.csv"):
        labels = row["single_purchase_fee_descr"].split(";")
        amounts = [cents(amount.split(" - ", 1)[0]) for amount in row["singe_purchase_fees"].split(";")]
        if labels != ["Installation Fee", "Deposit Fee", "Activation Fee", "Other One Time Fee"] or row["monthly_provider_fee_descr"] != "Router Lease":
            raise ValueError("Mitchell Telecom fee categories changed")
        fees = dict(zip(labels, amounts, strict=True))
        mitchell.append(plan(row, fees["Installation Fee"] + fees["Activation Fee"]))
        mitchell_all_fees.append(plan(row, sum(amounts)))

    nk = {}
    for name in SOURCES:
        if not name.startswith("nkt-"):
            continue
        row, = read_rows(folder / name)
        if cents(row["single_purchase_fees"]) or cents(row["monthly_provider_fee"]):
            raise ValueError(f"NKTELCO fee categories changed in {name}")
        nk[name[4:-4]] = (row["unique_plan_id"], plan(row, 0))
    if nk["500M"][0] != nk["Basic-500M"][0] or nk["1GIG"][0] != nk["Standard-1GIG"][0]:
        raise ValueError("The NKTELCO label pairs no longer match by ID")
    common = [nk[name][1] for name in ("2GIG", "5GIG")]
    alternatives = {
        "NKTELCO": ("Basic-500M", "Standard-1GIG"),
        "NKTELCO with alternative Basic quote": ("500M", "Standard-1GIG"),
        "NKTELCO with alternative Standard quote": ("Basic-500M", "1GIG"),
        "NKTELCO with both alternative quotes": ("500M", "1GIG"),
    }
    return {
        "GWI": [plan(row, cents(row["single_purchase_fees"])) for row in gwi],
        "Mitchell Telecom": mitchell,
        "Mitchell Telecom with additional upfront charges": mitchell_all_fees,
        **{name: [*common, nk[basic][1], nk[standard][1]] for name, (basic, standard) in alternatives.items()},
    }


def scenarios(name, offers):
    plans = tuple(sorted(set(offers)))
    for months, speed, monthly in product((12, 24, 36), (100, 400, 800), (45, 65, 85, 105)):
        case = Case(name, months, speed, 100 * monthly * months, plans)
        if sum(plan.speed >= speed for plan in plans) >= 2 and any(not case.value(plan, "total")[0] for plan in plans):
            yield case


def score(cases):
    outcomes = {regime: {"optimal": 0, "failures": 0} for regime in ("opaque", "disclosed", "total")}
    opportunities = 0
    for case in cases:
        candidates = [plan for plan in case.plans if plan.speed >= case.speed]
        opportunities += any((a.initial - b.initial) * (a.total(case.months) - b.total(case.months)) < 0
                             for a, b in combinations(candidates, 2))
        for regime, counts in outcomes.items():
            optimal, failed = case.outcome(case.pick(range(len(case.plans)), regime))
            counts["optimal"] += optimal
            counts["failures"] += failed
    return {"scenarios": len(cases), "strict_price_reversals": opportunities, "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("zito", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    panels = {provider: {"plans": len(offers), **score(list(scenarios(provider, offers)))}
              for provider, offers in portfolios(args.labels).items()}
    panel, *_ = tasks(args.zito)
    panels["Zito Media"] = score(panel)
    inputs = {name: {"url": url, "sha256": hashlib.sha256((args.labels / name).read_bytes()).hexdigest()}
              for name, url in SOURCES.items()}
    inputs["Zito Media"] = {"sha256": hashlib.sha256(args.zito.read_bytes()).hexdigest()}
    write_json(args.output, {"inputs": inputs, "providers": panels})


if __name__ == "__main__":
    main()
