"""Combine validated matched-panel summaries in the frozen checkpoint order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from scripts.collect_live import write_json
from scripts.model_breadth import caps


MODELS = {
    "qwen3-4b-q4": ("Qwen3 4B", "Qwen3"),
    "qwen3-8b-q4": ("Qwen3 8B", "Qwen3"),
    "qwen3-14b-q4": ("Qwen3 14B", "Qwen3"),
    "llama31-8b-q4": ("Llama 3.1 8B", "Llama"),
    "gemma-3-4b-q4": ("Gemma 3 4B", "Gemma 3"),
    "gemma3-12b-q4": ("Gemma 3 12B", "Gemma 3"),
    "mistral-7b-v03-q4": ("Mistral 7B", "Mistral"),
    "phi-4-mini-q8": ("Phi-4-mini", "Phi"),
    "gpt-oss-20b-mxfp4": ("gpt-oss 20B", "gpt-oss"),
    "granite33-8b-q4": ("Granite 3.3 8B", "Granite"),
    "olmo3-7b-q4": ("OLMo 3 7B", "OLMo"),
    "falcon3-7b-q4": ("Falcon3 7B", "Falcon3"),
    "aya8b-q4": ("Aya Expanse 8B", "Aya"),
    "smollm3-3b-q4": ("SmolLM3 3B", "SmolLM3"),
}
SUPPLEMENTARY_MODELS = {
    "llama32-3b-q4": ("Llama 3.2 3B", "Llama"),
    "mistral-small32-24b-q4": ("Mistral Small 3.2 24B", "Mistral"),
    "phi4-14b-q4": ("Phi-4 14B", "Phi"),
    "command-r7b-q4": ("Command R7B", "Command"),
    "gemma3-27b-q4": ("Gemma 3 27B", "Gemma 3"),
    "qwen3-30b-a3b-q4": ("Qwen3 30B-A3B", "Qwen3"),
}
SYNTHETIC_COUNTS = {"benign": 96, "decoy": 96, "decoy_injection": 96, "unconditioned": 128}


def highlights(rows):
    synthetic = {}
    for market in SYNTHETIC_COUNTS:
        outcomes = [row["synthetic"][market]["counts"] for row in rows]
        gains = [row["disclosed_optimal"] - row["opaque_optimal"] for row in outcomes]
        synthetic[market] = {
            "median_gain_points": 100 * median(gains) / SYNTHETIC_COUNTS[market],
            "checkpoints_improved": sum(gain > 0 for gain in gains),
            "checkpoints_fewer_failures": sum(row["disclosed_failure"] < row["opaque_failure"] for row in outcomes),
            "checkpoints_opaque_pool_below_one": sum(row["opaque_optimal"] < row["one_optimal"] for row in outcomes),
        }

    catalogs = {}
    pooled = [row["catalogs"]["pooled"] for row in rows]
    for parser in ("strict", "unfenced", "leading_json"):
        cases = pooled if parser == "strict" else [row[parser] for row in pooled]
        optima = [row["verified"]["optimal"] for row in cases]
        catalogs[parser] = {"headline_approvals": sum(row["opaque"]["approved"] for row in cases),
                            "over_budget_approvals": sum(row["opaque"]["unsafe"] for row in cases),
                            "compliant_retained": sum(row["retained_compliant"] for row in cases),
                            "verified_optima_range": [min(optima), max(optima)]}
    return {"synthetic": synthetic, "catalogs": catalogs}


def merge(directory):
    rows, fingerprints = [], set()
    for model, (label, family) in MODELS.items():
        item = json.loads((directory / model / "summary.json").read_text())
        synthetic, public = item["synthetic"], item["catalogs"]["pooled"]
        if item["model"] != model or (item["synthetic_cap"], item["catalog_cap"]) != caps(model):
            raise ValueError(f"Wrong checkpoint or request profile for {model}")
        if {market: section["tasks"] for market, section in synthetic.items()} != SYNTHETIC_COUNTS:
            raise ValueError(f"Incomplete synthetic panel for {model}")
        if (public["tasks"], public["feasible"]) != (96, 67):
            raise ValueError(f"Incomplete catalog panel for {model}")
        for parser in (public, public["unfenced"], public["leading_json"]):
            if parser["verified"]["unsafe"] != 0:
                raise ValueError(f"Invalid catalog gate replay for {model}")
            if parser["retained_compliant"] != parser["opaque"]["compliant"]:
                raise ValueError(f"The gate lost a previously eligible nomination for {model}")
            if parser["pool_optimal"] != parser["verified"]["optimal"]:
                raise ValueError(f"The gate lost an attainable catalog optimum for {model}")
        fingerprints.add((item["engine_revision"], item["frame_sha256"], item["corpus_sha256"]))
        rows.append({"label": label, "family": family, **item})
    if len(fingerprints) != 1:
        raise ValueError("The panel must use the same build revision and archived catalogs")
    return {"models": rows, "synthetic_tasks": SYNTHETIC_COUNTS, "catalog_tasks": 96, "catalog_feasible": 67,
            "highlights": highlights(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, merge(args.directory))


if __name__ == "__main__":
    main()
