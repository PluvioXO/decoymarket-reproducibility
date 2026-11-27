"""Count the effect of reusing identical catalog evidence across shopper tasks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.catalog_verification import lookup_order, tasks, validate, with_second_lookup
from scripts.collect_live import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = json.loads(args.frame.read_bytes())
    panels = validate(frame, json.loads(args.corpus.read_bytes()))
    _, test = tasks(frame, panels)
    separate = defaultdict(Counter)
    shared = defaultdict(lambda: defaultdict(set))
    for case in test.values():
        name = case[0]
        ordered = lookup_order(panels[name]["offers"], case)
        sources = list(dict.fromkeys(offer["source"] for offer in ordered))
        for rule, read in (("all_sources", sources), ("headline_with_stopping", with_second_lookup(sources[:1], ordered, case))):
            separate[name][rule] += len(read)
            shared[name][rule].update(read)
    write_json(args.output, {"test_tasks": len(test), "separate": separate,
                             "same_snapshot_cache": {name: {rule: len(keys) for rule, keys in counts.items()}
                                                     for name, counts in shared.items()}})


if __name__ == "__main__":
    main()
