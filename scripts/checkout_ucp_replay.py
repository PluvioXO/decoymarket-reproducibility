"""Exercise changed quotes on the UCP reference merchant with its mock payment handler."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from itertools import product
import json
import logging
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from scripts.collect_live import write_json


REVISION = "3a104e5ec8a6002a158e09add27f17ae417845e4"
EVENTS = ("unchanged", "surcharge", "lower_charge", "destination", "quantity", "substitution")
POLICIES = ("session_id", "exact_quote", "recheck_mandate")
BUDGETS = (4500, 6500, 8500)
PAYMENT = {"payment": {"instruments": [{"id": "sample", "handler_id": "mock_payment_handler", "type": "card",
                                       "display": {"brand": "Visa", "last_digits": "1234"},
                                       "credential": {"type": "token", "token": "success_token"}}]}, "risk_signals": {}}


def request(client, method, path, body=None):
    nonce = str(uuid4())
    response = client.request(method, path, json=body, headers={"UCP-Agent": "DecoyMarket sandbox fixture",
                              "request-id": nonce, "idempotency-key": nonce})
    if not response.is_success:
        raise RuntimeError(f"{method} {path.split('/')[1]} returned {response.status_code}: {response.text[:900]}")
    return response.json()


def update(client, sid, line_id, sku, quantity, zip_code, option):
    return request(client, "PUT", f"/checkout-sessions/{sid}", {
        "line_items": [{"id": line_id, "item": {"id": sku}, "quantity": quantity}],
        "fulfillment": {"methods": [{
            "id": "shipping", "type": "shipping", "line_item_ids": [line_id], "selected_destination_id": "dest",
            "destinations": [{"id": "dest", "address_country": "US", "postal_code": zip_code}],
            "groups": [{"id": "parcel", "line_item_ids": [line_id], "selected_option_id": option}],
        }]},
    })


def snapshot(quote):
    methods = quote["fulfillment"]["methods"]
    method, = methods
    dest, = [address for address in method["destinations"] if address["id"] == method["selected_destination_id"]]
    option, = [group["selected_option_id"] for group in method["groups"]]
    lines = tuple((line["item"]["id"], line["quantity"]) for line in quote["line_items"])
    total, = [row["amount"] for row in quote["totals"] if row["type"] == "total"]
    return {"currency": quote["currency"], "lines": lines, "location": (dest["address_country"], dest["postal_code"]),
            "shipping": option, "total": total, "charges": sorted((row["type"], row["amount"]) for row in quote["totals"])}


def case(client, sku, alternative, quantity, event, budget, policy):
    original = request(client, "POST", "/checkout-sessions", {
        "line_items": [{"item": {"id": sku}, "quantity": quantity}],
        "buyer": {"first_name": "Test", "email": "shopper@example.invalid"},
    })
    sid = original["id"].split("/")[-1]
    line_id = original["line_items"][0]["id"]
    earlier = snapshot(update(client, sid, line_id, sku, quantity, "10001",
                              "exp-ship-us" if event == "lower_charge" else "std-ship"))
    newer = snapshot(update(client, sid, line_id,
                             alternative if event == "substitution" else sku,
                             quantity + (event == "quantity"), "94103" if event == "destination" else "10001",
                             "exp-ship-us" if event == "surcharge" else "std-ship"))
    assert earlier["shipping"] == ("exp-ship-us" if event == "lower_charge" else "std-ship")
    assert newer["shipping"] == ("exp-ship-us" if event == "surcharge" else "std-ship")
    assert (earlier == newer) == (event == "unchanged")
    initial = earlier["total"] <= budget
    scope_kept = all(earlier[key] == newer[key] for key in ("currency", "lines", "location"))
    eligible = scope_kept and newer["total"] <= budget
    authorize = initial and (policy == "session_id" or (earlier == newer if policy == "exact_quote" else eligible))
    order_total = None
    if authorize:
        receipt = request(client, "POST", f"/checkout-sessions/{sid}/complete", PAYMENT)
        assert receipt["status"] == "completed" and snapshot(receipt) == newer
        order = request(client, "GET", f"/orders/{receipt['order']['id']}")
        order_total, = [row["amount"] for row in order["totals"] if row["type"] == "total"]
        assert order_total == newer["total"]
    metrics = {"eligible_initial": initial, "eligible_latest": initial and eligible, "completed": authorize,
            "stale_completed": authorize and earlier != newer, "violation": authorize and not eligible,
            "valid_completion": authorize and eligible, "referred": initial and not authorize,
            "authorized_total": earlier["total"], "committed_total": order_total or 0}
    return metrics, earlier, newer


def run(reference):
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected UCP reference {REVISION}, found {revision}")
    server = reference / "rest/python/server"
    fixtures = server.parent / "test_data/flower_shop"
    sys.path[:0] = [str(server.parent), str(server)]
    from absl import flags
    from fastapi.testclient import TestClient
    from server.server import app

    logging.getLogger().setLevel(logging.ERROR)
    with (fixtures / "products.csv").open() as handle:
        products = [row["id"] for row in csv.DictReader(handle)]
    counts, quotes = defaultdict(Counter), []
    with TemporaryDirectory(prefix="decoy-ucp-") as work:
        db = Path(work) / "transactions.db"
        db_args = [f"--products_db_path={work}/products.db", f"--transactions_db_path={db}"]
        subprocess.run([sys.executable, str(server / "import_csv.py"), *db_args, f"--data_dir={fixtures}"],
                       check=True, capture_output=True)
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("UPDATE inventory SET quantity = 10000")
        flags.FLAGS(["decoy-ucp", *db_args])
        with TestClient(app) as client:
            for i, quantity, event, budget, policy in product(range(len(products)), (1, 2), EVENTS, BUDGETS, POLICIES):
                result, earlier, newer = case(client, products[i], products[(i + 1) % len(products)], quantity, event, budget, policy)
                groups = (policy, f"{policy}/{event}")
                for group in groups:
                    counts[group].update(cases=1, **result)
                if policy == "session_id" and result["eligible_initial"]:
                    quotes.append({"sku": products[i], "quantity": quantity, "event": event, "budget_cents": budget,
                                   "before": earlier, "after": newer, "within_mandate": result["eligible_latest"]})
    assert counts["exact_quote"]["stale_completed"] == counts["recheck_mandate"]["violation"] == 0
    assert counts["session_id"]["eligible_initial"] == counts["session_id"]["completed"]
    return {"reference_revision": revision, "payment": "reference mock handler", "budgets_cents": BUDGETS,
            "products": len(products), "counts": counts, "quotes": quotes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, run(args.reference.resolve()))


if __name__ == "__main__":
    main()
