"""Command line for DecoyMarket: generate, example, run, report, smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import generate_task, optimum
from .prompts import drafter_prompt
from .run import RunConfig, write_episode


def _seeds(spec: str):
    """'start:count' or 'a,b,c' -> list of ints."""
    if ":" in spec:
        start, count = spec.split(":")
        return list(range(int(start), int(start) + int(count)))
    return [int(x) for x in spec.split(",")]


def cmd_example(args):
    t = generate_task(args.seed, args.market)
    print(drafter_prompt(t)[1]["content"])
    print("\n# optimum:", optimum(t), "decoys:", t.decoy_sellers)


def cmd_generate(args):
    seeds = _seeds(args.seeds)
    out = {s: generate_task(s, args.market).to_json() for s in seeds}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} {args.market} tasks to {args.out}")


def cmd_run(args):
    cfg = RunConfig(
        model_name=args.model_name, market=args.market, seeds=_seeds(args.seeds),
        backend=args.backend, n_drafts=args.n_drafts, max_tokens=args.max_tokens,
        endpoint=args.endpoint, model_id=args.model_id, run_seed=args.run_seed,
        token_budget=args.token_budget,
    )
    ep = write_episode(cfg, args.out)
    print(f"wrote {args.out}: {len(ep['tasks'])} tasks, backend={ep['backend']}, "
          f"tokens={ep['total_tokens']}, mock={ep['is_mock']}")


def cmd_report(args):
    from . import report
    analyses = report.build(args.results, args.generated)
    for m, res in analyses.items():
        print(f"{m}: P1={res['p1']:.3f} peak(k={res['peak_k']})={res['peak']:.3f} "
              f"P16={res['p16']:.3f} disclosed_P16={res['p16_disclosed']:.3f} "
              f"rates_mae={res['forecast']['rates_mae']:.3f}")
    print(f"wrote generated LaTeX to {args.generated}")


def cmd_smoke(args):
    """Fabricate mock episodes for all markets (cal+test) and build the report. Not a model."""
    from . import report
    rd = Path(args.results)
    rd.mkdir(parents=True, exist_ok=True)
    rates = {
        "benign": {"S": 0.55, "A": 0.0, "B": 0.25, "N": 0.2},
        "decoy": {"S": 0.4, "A": 0.4, "B": 0.1, "N": 0.1},
        "decoy_injection": {"S": 0.33, "A": 0.5, "B": 0.1, "N": 0.07},
    }
    for market in ("benign", "decoy", "decoy_injection"):
        for split, seeds in (("cal", range(1, 41)), ("test", range(1001, 1081))):
            out = rd / f"{market}_{split}.json"
            if out.exists():
                out.unlink()
            cfg = RunConfig(model_name="mock-stub", market=market, seeds=list(seeds),
                            backend="mock", n_drafts=16, run_seed=7, mock_rates=rates[market])
            write_episode(cfg, str(out))
    report.build(str(rd), args.generated)
    print(f"[smoke] mock episodes in {rd}, generated LaTeX in {args.generated}")
    print("[smoke] all numbers are fabricated by the mock stub; software check only.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="decoymarket")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("example", help="print one task's drafter prompt")
    pe.add_argument("--seed", type=int, default=3)
    pe.add_argument("--market", default="decoy")
    pe.set_defaults(func=cmd_example)

    pg = sub.add_parser("generate", help="write a task set to JSON")
    pg.add_argument("--market", default="decoy")
    pg.add_argument("--seeds", default="1:100")
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=cmd_generate)

    pr = sub.add_parser("run", help="run one (model, market) cell to an episode log")
    pr.add_argument("--backend", choices=["mock", "live"], default="mock")
    pr.add_argument("--model-name", required=True)
    pr.add_argument("--market", required=True)
    pr.add_argument("--seeds", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--n-drafts", type=int, default=16)
    pr.add_argument("--max-tokens", type=int, default=1024)
    pr.add_argument("--endpoint", default="")
    pr.add_argument("--model-id", default="")
    pr.add_argument("--run-seed", type=int, default=0)
    pr.add_argument("--token-budget", type=int, default=0)
    pr.set_defaults(func=cmd_run)

    prp = sub.add_parser("report", help="build generated LaTeX from a results directory")
    prp.add_argument("--results", required=True)
    prp.add_argument("--generated", default="paper/generated")
    prp.set_defaults(func=cmd_report)

    ps = sub.add_parser("smoke", help="fabricate mock episodes and build the report (software check)")
    ps.add_argument("--results", default="results/mock")
    ps.add_argument("--generated", default="paper/generated")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
