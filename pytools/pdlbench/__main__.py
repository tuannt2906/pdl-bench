"""Command-line entry point: ``python -m pdlbench <subcommand>`` (or ``pdlbench`` once installed).

Subcommands:
    data       download MNIST / CIFAR-10 and convert to .pdlds
    env        capture the host environment as JSON
    run        run an experiment described by a YAML config
    aggregate  flatten + summarise results into CSV / LaTeX tables
    plot       produce figures (PNG + PDF)
    perfmodel  fit the α-β model and compare predictions with measurements
    report     write a Markdown report
    ai-eval    source-code metrics for the AI coding-agent evaluation
    all        run + aggregate + plot + perfmodel + report for one config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, aggregate, codemetrics, data, envinfo, perfmodel, plots, report, runner


def _cmd_all(args: argparse.Namespace) -> int:
    run_dir = runner.run_experiment(Path(args.config), Path(args.build_dir), Path(args.results_dir),
                                    repeats=args.repeats, filters=args.filter)
    if run_dir is None:
        return 1
    perfmodel.analyze(run_dir, Path(args.comm_dir) if args.comm_dir else None)
    a = aggregate.analyze(run_dir)
    aggregate.write(a)
    plots.plot_all(a)
    out = run_dir / "report.md"
    out.write_text(report.build_report([run_dir], run_dir.resolve()), encoding="utf-8")
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdlbench", description="pdl-bench experiment & analysis tooling")
    p.add_argument("--version", action="version", version=f"pdlbench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    for name, mod, help_ in (
        ("data", data, "download and convert datasets to .pdlds"),
        ("env", envinfo, "capture host environment as JSON"),
        ("run", runner, "run an experiment YAML"),
        ("aggregate", aggregate, "aggregate results into CSV/LaTeX tables"),
        ("plot", plots, "plot figures for run directories"),
        ("perfmodel", perfmodel, "fit and evaluate performance models"),
        ("report", report, "generate a Markdown report"),
        ("ai-eval", codemetrics, "code metrics for AI coding-agent evaluation"),
    ):
        sp = sub.add_parser(name, help=help_, description=(mod.__doc__ or "").strip().splitlines()[0])
        mod.add_arguments(sp)
        sp.set_defaults(func=mod.main)
    sp = sub.add_parser("all", help="run + aggregate + plot + perfmodel + report")
    sp.add_argument("config", help="experiment YAML")
    sp.add_argument("--build-dir", default="build")
    sp.add_argument("--results-dir", default="results")
    sp.add_argument("--repeats", type=int)
    sp.add_argument("--filter", action="append", metavar="KEY=VAL")
    sp.add_argument("--comm-dir", help="comm benchmark run dir for the α-β model")
    sp.set_defaults(func=_cmd_all)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
