"""Capture official source documents used in the catalog replay and record content digests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import monotonic
from urllib.parse import urldefrag

from scripts.collect_live import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    urls = sorted({urldefrag(url).url for panel in corpus["catalogs"].values()
                   for url in (panel["url"], *panel["sources"].values())} | set(corpus["evidence"]))
    args.output.mkdir(parents=True, exist_ok=True)

    def capture(item):
        order, url = item
        name = f"source-{order:03d}.bin"
        path = args.output / name
        start = monotonic()
        response = subprocess.run(["curl", "-sSL", "--max-time", "30", "-o", str(path),
                                   "-w", "%{http_code}|%{url_effective}", url], capture_output=True, text=True)
        status, _, effective = response.stdout.partition("|")
        record = {"url": url, "effective": effective, "status": status, "rc": response.returncode,
                  "seconds": monotonic() - start, "captured": datetime.now(timezone.utc).isoformat()}
        if response.returncode == 0 and status == "200":
            data = path.read_bytes()
            record.update(file=name, bytes=len(data), sha256=sha256(data).hexdigest())
        else:
            path.unlink(missing_ok=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(capture, enumerate(urls, 1)))
    write_json(args.output / "manifest.json", records)
    print(f"Captured {sum('file' in record for record in records)}/{len(records)} source documents")


if __name__ == "__main__":
    main()
