"""Capture public product cards and bound cart shipping estimates for a frozen frame."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

from scripts.collect_live import write_json


def fetch(client, url, evidence, payload=None):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(payload).encode()
    with client.open(Request(url, data=payload, headers=headers), timeout=15) as response:
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("oversize response")
    evidence.append({"url": url, "sha256": sha256(data).hexdigest(),
                     "captured_utc": datetime.now(timezone.utc).isoformat()})
    return json.loads(data)


def reason(error):
    if isinstance(error, HTTPError):
        return f"http-{error.code}"
    if isinstance(error, json.JSONDecodeError):
        return "non-json response"
    if isinstance(error, (URLError, TimeoutError, OSError)):
        return "network failure"
    return str(error) if isinstance(error, ValueError) else "incomplete response"


def cart_quote(origin, variant, quantity, address, *, client=None):
    evidence = []
    stage = "cart_add"
    try:
        client = client or build_opener(HTTPCookieProcessor(CookieJar()))
        fetch(client, origin + "/cart/add.js", evidence, {"items": [{"id": variant, "quantity": quantity}]})
        stage = "cart"
        cart = fetch(client, origin + "/cart.js", evidence)
        lines = cart["items"]
        if (cart["currency"] != "USD" or cart.get("taxes_included") or len(lines) != 1
                or (lines[0]["variant_id"], lines[0]["quantity"]) != (variant, quantity)
                or lines[0].get("selling_plan_allocation")):
            raise ValueError("unmatched cart line, currency, or terms")
        subtotal = cart["total_price"]
        if type(subtotal) is not int or subtotal < 0:
            raise ValueError("invalid cart merchandise total")
        stage = "shipping"
        query = urlencode({f"shipping_address[{key}]": value for key, value in address.items()})
        data = fetch(client, origin + "/cart/shipping_rates.json?" + query, evidence)
        rates = []
        for item in data["shipping_rates"]:
            price = Decimal(str(item["price"])) * 100
            if not price.is_finite() or price != price.to_integral_value() or price < 0 or item.get("currency", "USD") != "USD":
                raise ValueError("invalid shipping amount or currency")
            rates.append({"name": item["name"], "cents": int(price)})
        if not rates:
            raise ValueError("no delivery rates")
        shipping = min(rate["cents"] for rate in rates)
        return {"stage": "matched", "variant": variant, "quantity": quantity, "currency": "USD",
                "subtotal_cents": subtotal, "shipping_cents": shipping, "total_cents": subtotal + shipping,
                "delivery_rates": rates, "evidence": evidence}
    except (ValueError, KeyError, TypeError, OSError, InvalidOperation) as error:
        return {"stage": stage, "reason": reason(error), "evidence": evidence}


def capture(offer, quantities, address):
    row = {**offer, "card": None, "quotes": {}, "product_evidence": []}
    url = urlsplit(offer["url"])
    origin = f"{url.scheme}://{url.netloc}"
    try:
        if url.scheme != "https" or not url.path.startswith("/products/"):
            raise ValueError("unsupported product URL")
        product = fetch(build_opener(), origin + url.path.rstrip("/") + ".js", row["product_evidence"])
        variants = product["variants"]
        if "variant" in offer:
            variant = next(v for v in variants if v["id"] == offer["variant"])
        elif "size_oz" in offer:
            sizes = "|".join(map(str, offer["size_oz"]))
            variant = min((v for v in variants if re.search(rf"\b(?:{sizes})\s*(?:fl\s*)?oz\b", v["title"], re.I)),
                          key=lambda v: v["id"])
        else:
            variant = variants[0]
        if type(variant["price"]) is not int or variant["price"] < 0 or type(variant["available"]) is not bool:
            raise ValueError("invalid product variant")
        row["card"] = {"title": product["title"], "variant_title": variant["title"], "variant": variant["id"],
                       "unit_cents": variant["price"], "available": variant["available"]}
        row["status"] = "card" if variant["available"] else "out_of_stock"
    except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError) as error:
        row.update(status="card_unresolved", reason=reason(error))
        return row
    if variant["available"]:
        row["quotes"] = {str(q): cart_quote(origin, variant["id"], q, address) for q in quantities}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.protocol.read_bytes()
    protocol = json.loads(data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(capture, offer, protocol["quantities"], protocol["shipping"]) for offer in protocol["offers"]]
        offers = [future.result() for future in futures]
    write_json(args.output, {"protocol_sha256": sha256(data).hexdigest(), "offers": offers})
    for offer in offers:
        print(offer["id"], offer["status"], {k: q["stage"] for k, q in offer["quotes"].items()}, flush=True)


if __name__ == "__main__":
    main()
