"""Freeze paired customer-facing broadband labels and publisher reference files."""
from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.broadband_transfer import cents
from scripts.collect_live import write_json
from scripts.tariff_census import read_rows


PAGES = {
    "nktelco": "https://www.nktelco.com/cxsr-labels/",
    "mitchell": "https://mitchelltelecom.com/residential-broadband-nutrition-labels/",
    "fpuan": "https://fpuanet.com/broadband-consumer-labels/",
}
TABLES = {
    "mitchell.csv": "https://mitchelltelecom.com/wp-content/uploads/2026/03/BroadbandLabels-RES-2026.csv",
    "fpuan.csv": "https://app.broadbandconsumerlabels.com/downloads/5ac3a475-3dbb-4548-9156-4c826dca7fe9/labels.csv",
}
FPU_CARDS = {
    "100 Mbps Standard": "Standard100",
    "500 Mbps Premium": "Premium500",
    "1 GIG Connect": "1GigConnect",
    "2 GIG Connect Plus": "2GigConnect",
    "5 GIG Ultra": "5GigConnect",
}
FIELDS = ("initial_cents", "intro_months", "regular_cents", "installation_cents", "activation_cents", "download_kbps", "contract_months")
PLAN_ID = re.compile(r"\bF\d{10}[A-Z0-9]{15}\b")
NKT_TABLE = re.compile(r'href="(https://nktelco\.com/wp-content/uploads/[^"?#]+\.(?:csv|xlsx))"')
ARTICLE = re.compile(r'<article class="bbl"[^>]*>.*?</article>', re.S)
NUM = r"[0-9][\d,.]*"


class VisibleText(HTMLParser):
    """Flatten the published text layer while leaving headings and table rows legible."""

    SKIP = {"head", "script", "style", "svg", "template", "noscript"}
    BREAK = {"article", "div", "section", "p", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt"}

    def __init__(self):
        super().__init__()
        self.skipped = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipped.append(tag)
        if not self.skipped and tag in self.BREAK:
            self.parts.append("\n")
        if not self.skipped and tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.skipped and tag == self.skipped[-1]:
            self.skipped.pop()
        if not self.skipped and tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipped:
            self.parts.append(re.sub(r"\s+", " ", data))


def plain(html):
    parser = VisibleText()
    parser.feed(html)
    return "\n".join(text for line in "".join(parser.parts).splitlines() if (text := " ".join(line.split())))


def read_xlsx(path):
    # shortcut: only the publisher's one-sheet, shared-string workbooks, extend for other formats.
    def values(row, strings):
        return {re.match(r"[A-Z]+", cell.attrib["r"])[0]: strings[int(value.text)] if cell.get("t") == "s" else value.text
                for cell in row if (value := cell.find("{*}v")) is not None}

    with ZipFile(path) as book:
        strings = ["".join(node.itertext()) for node in ElementTree.fromstring(book.read("xl/sharedStrings.xml"))]
        header, data = [row for row in ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//{*}row")
                        if row.find(".//{*}v") is not None]
    by_column = values(data, strings)
    return [{name.strip(): value.strip() if value.strip() != "NULL" else ""
             for column, name in values(header, strings).items() if (value := by_column.get(column)) is not None}]


def price(value):
    match = re.search(NUM, value)
    if not match:
        raise ValueError(f"Expected a quoted price: {value!r}")
    return cents(match[0])


def speed(value):
    match = re.fullmatch(rf"({NUM})\s*(Mbps|Gbps)?", value.strip(), re.I)
    if not match:
        raise ValueError(f"Unexpected download speed: {value!r}")
    scale = 1_000_000 if (match[2] or "").lower() == "gbps" else 1_000
    return int(Decimal(match[1].replace(",", "")) * scale)


def machine(row, provider):
    if provider == "fpuan":
        if (row["has_intro_price"], row["contract_required"], row["contract_duration"]) != ("false", "true", "Month"):
            raise ValueError("Unexpected FPUAnet price terms")
        monthly = price(row["monthly_price"])
        return [monthly, 0, monthly, None, None, speed(row['download_speed'] + ' ' + row['download_speed_unit']), int(row['contract_length'])]
    regular = price(row["monthly_price"])
    if row["intro_rate"] not in ("Yes", "No") or row["contract_req"] != "No":
        raise ValueError(f"Unexpected published contract or intro terms in {row['unique_plan_id']}")
    has_intro = row["intro_rate"] == "Yes"
    descriptions = row.get("single_purchase_fee_descr", "").split(";")
    amounts = row.get("single_purchase_fees", row.get("singe_purchase_fees", "")).split(";")
    fees = dict(zip(descriptions, amounts, strict=True))
    install = sum(price(value) for key, value in fees.items() if re.match(r"install(?:ation)? fee\b", key, re.I))
    activation = sum(price(value) for key, value in fees.items() if key.lower().startswith("activation fee"))
    return [price(row["intro_rate_price"]) if has_intro else regular,
            int(row["intro_rate_time"].split()[0]) if has_intro else 0, regular, install,
            activation, speed(row["typical_download_speed"]), 0]


def display(label):
    flat = " ".join(label.split())

    def required(pattern):
        match = re.search(pattern, flat, re.I)
        if not match:
            raise ValueError(f"Missing {pattern!r} in {label[:140]!r}")
        return match[1]

    def fee(term):
        # The charged amount follows any parenthetical reference price, such as a waived regular installation fee.
        match = re.search(rf"\b{term}\s*(?:\([^)]*\)\s*)?\$\s*({NUM})", flat, re.I)
        return price(match[1]) if match else 0

    start = price(required(rf"\bmonthly price\s*\$\s*({NUM})"))
    intro = re.search(r"(?:monthly price is an introductory rate\??\s*)(yes|no)|introductory price\s+(none)", flat, re.I)
    if not intro:
        raise ValueError(f"Missing introductory status in {label[:140]!r}")
    promotional = (intro[1] or "").lower() == "yes"
    months = int(required(r"time the introductory rate applies\s*(\d+)\s*months?")) if promotional else 0
    regular = price(required(rf"monthly price after the introductory rate\s*\$\s*({NUM})")) if promotional else start
    commitment = re.search(r"(?:length of contract|contract duration)\s*(N/A|none|no|\d+)", flat, re.I)
    contract = None if commitment is None else int(commitment[1]) if commitment[1].isdigit() else 0
    down = speed(required(rf"typical download speed\s*({NUM}\s*(?:mbps|gbps))"))
    return [start, months, regular, fee(r"(?:one-time )?(?:installation|install) fee"),
            fee("activation fee"), down, contract]


def label_div(card):
    start = card.index('<div class="bb-label">')
    depth = 0
    for match in re.finditer(r"<div\b[^>]*>|</div\s*>", card[start:], re.I):
        depth += -1 if match[0].startswith("</") else 1
        if depth == 0:
            return card[start:start + match.end()]
    raise ValueError("FPUAnet label did not close")


def build(folder):
    inputs = {}

    def source(name, url):
        path = folder / name
        inputs[name] = {"url": url, "sha256": sha256(path.read_bytes()).hexdigest()}
        return path

    output = {"sources": inputs, "pages": {}, "cases": {}}
    for provider, url in PAGES.items():
        html = source(provider + ".html", url).read_text()
        output["pages"][provider] = plain(html)
        source_rows = {}
        if provider in ("mitchell", "fpuan"):
            table = source(provider + ".csv", TABLES[provider + ".csv"])
            source_rows = {row["unique_plan_id"]: row for row in read_rows(table)}
        if provider == "fpuan":
            cards = html.split('<div class="bbl-label">')[1:]
            panels = [(plain(label_div(card)), plain(re.search(r'<div class="bbl-plan-name">(.*?)</div>', card, re.S)[1]), None)
                      for card in cards]
            if {title for _, title, _ in panels} != FPU_CARDS.keys():
                raise ValueError("Unmatched public FPUAnet cards")
        else:
            matches = list(ARTICLE.finditer(html))
            panels = []
            for index, match in enumerate(matches):
                label = plain(match[0])
                ids = PLAN_ID.findall(label)
                if len(ids) != 1:
                    raise ValueError(f"Expected one plan ID in {provider} label {index + 1}")
                link = None
                if provider == "nktelco":
                    stop = matches[index + 1].start() if index + 1 < len(matches) else len(html)
                    link, = NKT_TABLE.findall(html[match.end():stop])
                panels.append((label, ids[0], link))
        for ordinal, (label, title, link) in enumerate(panels, 1):
            identity = FPU_CARDS[title] if provider == "fpuan" else title
            if link:
                table = source(Path(urlsplit(link).path).name, link)
                row, = read_xlsx(table) if table.suffix == ".xlsx" else read_rows(table)
            else:
                row = source_rows[identity]
            if row["unique_plan_id"] != identity:
                raise ValueError(f"Unmatched machine-readable record for {provider} label {ordinal}")
            if label not in output["pages"][provider]:
                raise ValueError(f"Label text cannot be found in {provider} page")
            output["cases"][f"{provider}-{ordinal:02}"] = {"provider": provider, "ordinal": ordinal, "label": label,
                "title": title, "machine_id": identity, "display": display(label), "machine": machine(row, provider)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        args.inputs.mkdir(parents=True, exist_ok=True)
        sources = {**{name + ".html": url for name, url in PAGES.items()}, **TABLES}
        for name, url in sources.items():
            with urlopen(url, timeout=30) as response:
                (args.inputs / name).write_bytes(response.read())
        for url in NKT_TABLE.findall((args.inputs / "nktelco.html").read_text()):
            with urlopen(unescape(url), timeout=30) as response:
                (args.inputs / Path(urlsplit(url).path).name).write_bytes(response.read())
    write_json(args.output, build(args.inputs))


if __name__ == "__main__":
    main()
