"""Capture public product variants and comparable cart estimates across ZIP codes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.cookiejar import CookieJar
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

from scripts.collect_live import write_json
from scripts.retail_cart_capture import capture, cart_quote, fetch, reason


def collect(protocol, output):
    frame = json.loads(protocol)

    def gather(offer):
        row = capture(offer, [], {})
        url = urlsplit(offer["url"])
        origin = f"{url.scheme}://{url.netloc}"
        card = row["card"]
        quotes = {}
        for zip_code, province in frame["locations"].items():
            for quantity in frame["quantities"]:
                key = f"{zip_code}-{quantity}"
                if card is None or not card["available"]:
                    quotes[key] = {"stage": row["status"], "reason": row.get("reason", "product unavailable")}
                    continue
                if any(quote.get("reason") == "http-429" for quote in quotes.values()):
                    quotes[key] = {"stage": "rate_limited", "reason": "http-429"}
                    continue
                began = monotonic()
                address = {"zip": zip_code, "country": "United States", "province": province}
                quote = cart_quote(origin, card["variant"], quantity, address)
                quotes[key] = {**quote, "elapsed_s": monotonic() - began}
        row["quotes"] = quotes
        print(row["id"], row["status"], sum(q["stage"] == "matched" for q in quotes.values()), len(quotes), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(gather, frame["offers"]))
    assert [r["id"] for r in rows] == [o["id"] for o in frame["offers"]]
    assert all(len(row["quotes"]) == len(frame["locations"]) * len(frame["quantities"]) for row in rows)
    write_json(output, {"protocol_sha256": sha256(protocol).hexdigest(), "offers": rows})


def retry_limited(protocol, previous, output):
    frame, data = json.loads(protocol), json.loads(previous)
    assert data["protocol_sha256"] == sha256(protocol).hexdigest()
    by_store = defaultdict(list)
    for row in data["offers"]:
        by_store[urlsplit(row["url"]).netloc].append(row)

    def recover(store_rows):
        for row in store_rows:
            limited = [scope for scope, quote in row["quotes"].items() if quote.get("reason") == "http-429"]
            if not limited:
                continue
            latest = capture({**row, "variant": row["card"]["variant"]}, [], {})
            if latest.get("reason") == "http-429":
                return
            if latest["card"] != row["card"]:
                row["retry_status"] = "original card could not be confirmed unchanged"
                continue
            origin = "https://" + urlsplit(row["url"]).netloc
            for scope in sorted(limited, key=lambda scope: sha256((row["id"] + scope).encode()).digest()):
                sleep(40)
                zip_code, quantity = scope.split("-")
                address = {"zip": zip_code, "country": "United States", "province": frame["locations"][zip_code]}
                began = monotonic()
                quote = cart_quote(origin, row["card"]["variant"], int(quantity), address)
                row["quotes"][scope] = {**quote, "elapsed_s": monotonic() - began, "previous_stage": row["quotes"][scope]["stage"]}
                print(row["id"], scope, quote["stage"], quote.get("reason", ""), flush=True)
                if quote.get("reason") == "http-429":
                    return

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(recover, by_store.values()))
    write_json(output, {**data, "first_capture_sha256": sha256(previous).hexdigest()})


def storefront_session(url, localized):
    u = urlsplit(url)
    client = build_opener(HTTPCookieProcessor(CookieJar()))
    if localized:
        form = urlencode({"form_type": "localization", "_method": "put", "country_code": "US", "return_to": u.path})
        request = Request(f"{u.scheme}://{u.netloc}/localization", data=form.encode(),
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        with client.open(request, timeout=15) as response:
            u = urlsplit(response.geturl())
    if u.scheme != "https" or u.netloc != urlsplit(url).netloc or "/products/" not in u.path:
        raise ValueError("unmatched localized product route")
    prefix = u.path.split("/products/", 1)[0]
    return client, f"{u.scheme}://{u.netloc}{prefix}", f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}.js"


def confirm_us(protocol, previous, output):
    frame, data = json.loads(protocol), json.loads(previous)
    assert data["protocol_sha256"] == sha256(protocol).hexdigest()

    def confirm(row):
        card = row["card"]
        if not card or not card["available"]:
            return row
        evidence = []
        try:
            client, base, product = storefront_session(row["url"], False)
            currency = fetch(client, base + "/cart.js", evidence)["currency"]
            localized = currency != "USD" or not any(q["stage"] == "matched" for q in row["quotes"].values())
            if localized:
                client, base, product = storefront_session(row["url"], True)
                currency = fetch(client, base + "/cart.js", evidence)["currency"]
            if currency != "USD":
                raise ValueError("USD presentation could not be confirmed")
            variant = next(v for v in fetch(client, product, evidence)["variants"] if v["id"] == card["variant"])
            if type(variant["price"]) is not int or variant["price"] < 0 or variant["available"] is not True:
                raise ValueError("variant unavailable in the USD storefront")
            if variant["price"] != card["unit_cents"] and not localized:
                raise ValueError("USD card changed after the previous capture")
        except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
            print(row["id"], "currency-unconfirmed", reason(error), flush=True)
            return {**row, "currency_status": reason(error), "currency_evidence": evidence}
        row = {**row, "currency_status": "USD", "currency_evidence": evidence,
               "card": {**card, "unit_cents": variant["price"], "variant_title": variant["title"]}}
        if localized:
            quotes = {}
            for scope in sorted(row["quotes"], key=lambda scope: sha256((row["id"] + scope).encode()).digest()):
                if any(quote.get("reason") == "http-429" for quote in quotes.values()):
                    quotes[scope] = {"stage": "rate_limited", "reason": "http-429"}
                    continue
                sleep(35)
                zip_code, quantity = scope.split("-")
                address = {"zip": zip_code, "country": "United States", "province": frame["locations"][zip_code]}
                began = monotonic()
                try:
                    fresh, route, _ = storefront_session(row["url"], True)
                    quote = cart_quote(route, card["variant"], int(quantity), address, client=fresh)
                except (ValueError, OSError) as error:
                    quote = {"stage": "localization", "reason": reason(error)}
                quotes[scope] = {**quote, "elapsed_s": monotonic() - began}
                print(row["id"], scope, quote["stage"], quote.get("reason", ""), flush=True)
            row["quotes"] = quotes
            if not any(q["stage"] == "matched" for q in quotes.values()):
                row["currency_status"] = "no bound USD quote after selecting US"
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(confirm, data["offers"]))
    write_json(output, {**data, "usd_confirmation": True, "prior_capture_sha256": sha256(previous).hexdigest(), "offers": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument("--retry-limited", type=Path)
    followup.add_argument("--confirm-us", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = args.protocol.read_bytes()
    if args.retry_limited:
        retry_limited(protocol, args.retry_limited.read_bytes(), args.output)
    elif args.confirm_us:
        confirm_us(protocol, args.confirm_us.read_bytes(), args.output)
    else:
        collect(protocol, args.output)


if __name__ == "__main__":
    main()
