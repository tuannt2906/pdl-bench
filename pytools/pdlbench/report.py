"""Markdown report generation from one or more analysed run directories."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import aggregate
from .util import fmt_value, human_bytes, load_json, log, resolve_run_dirs

ENV_KEYS = [("hostname", "Host"), ("os", "OS"), ("cpu_model", "CPU"), ("physical_cores", "Physical cores"),
            ("logical_cpus", "Logical CPUs"), ("memory_bytes", "Memory"), ("nvidia_smi", "GPU"),
            ("gcc", "Compiler"), ("nvcc", "CUDA"), ("mpirun", "MPI"), ("cmake", "CMake"), ("python", "Python")]


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        return f"{v:.{digits}g}"
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def md_table(df: pd.DataFrame, cols: Sequence[str], headers: Optional[Sequence[str]] = None,
             max_rows: int = 60) -> str:
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return "_no data_\n"
    hdr = list(headers) if headers and len(headers) == len(cols) else cols
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(cols)]
    for _, r in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows in the CSV files_")
    return "\n".join(lines) + "\n"


def _env_section(manifest: Dict[str, Any], train: pd.DataFrame) -> str:
    env = manifest.get("environment") or {}
    rows = []
    for key, label in ENV_KEYS:
        v = env.get(key)
        if key == "memory_bytes" and v:
            v = human_bytes(float(v))
        if v:
            rows.append(f"| {label} | {_fmt(v)} |")
    git = env.get("git") or {}
    if git.get("commit"):
        rows.append(f"| Git commit | `{git['commit'][:12]}`{' (dirty)' if git.get('dirty') else ''} |")
    if not train.empty and "host_gpu_name" in train.columns:
        rows.append(f"| GPU (binary) | {train['host_gpu_name'].dropna().iloc[0] if train['host_gpu_name'].notna().any() else '–'} |")
    if not rows:
        return "_environment not recorded_\n"
    return "| Item | Value |\n|---|---|\n" + "\n".join(rows) + "\n"


def _findings(a: aggregate.Analysis) -> List[str]:
    out: List[str] = []
    s = a.get("train_summary")
    if not s.empty:
        if "speedup" in s.columns and s["speedup"].notna().any():
            best = s.loc[s["speedup"].idxmax()]
            out.append(f"Best speedup **{best['speedup']:.2f}×** at {aggregate_label(best, a.group_cols)} "
                       f"(efficiency {_fmt(best.get('efficiency'), 3)}).")
            if "np" in s.columns:
                top = s[pd.to_numeric(s["np"], errors="coerce") == pd.to_numeric(s["np"], errors="coerce").max()]
                if "efficiency" in top.columns and top["efficiency"].notna().any():
                    out.append(f"At the largest process count (p={int(top['np'].iloc[0])}) parallel efficiency "
                               f"ranges {top['efficiency'].min():.2f}–{top['efficiency'].max():.2f}.")
        if "comm_fraction_mean" in s.columns and "np" in s.columns and s["comm_fraction_mean"].notna().any():
            g = s.groupby("np")["comm_fraction_mean"].mean().sort_index()
            trend = ", ".join(f"p={int(k)}: {v * 100:.1f}%" for k, v in g.items())
            out.append(f"Communication + synchronization share of training time: {trend}.")
        if "load_imbalance_mean" in s.columns and s["load_imbalance_mean"].notna().any():
            out.append(f"Maximum load imbalance (max/mean compute): {s['load_imbalance_mean'].max():.2f}.")
        if "final_test_acc_mean" in s.columns and s["final_test_acc_mean"].notna().any():
            best = s.loc[s["final_test_acc_mean"].idxmax()]
            out.append(f"Highest final test accuracy **{best['final_test_acc_mean'] * 100:.2f}%** with "
                       f"{aggregate_label(best, a.group_cols)} in {best['train_time_s_mean']:.1f}s.")
        tta = time_to_accuracy(a)
        if tta is not None and not tta.empty:
            fastest = tta.sort_values("time_to_target_s").iloc[0]
            out.append(f"Fastest to {tta['target'].iloc[0] * 100:.1f}% test accuracy: {fastest['config']} "
                       f"({fastest['time_to_target_s']:.1f}s).")
    ar = a.get("comm_allreduce_summary")
    if not ar.empty:
        big = ar[ar["bytes"] == ar["bytes"].max()]
        if not big.empty:
            best = big.loc[big["time_mean_s_mean"].idxmin()]
            out.append(f"Fastest allreduce for {human_bytes(float(best['bytes']))}: **{best['algo']}** "
                       f"(p={int(best['world_size'])}, {best['time_mean_s_mean'] * 1e3:.2f} ms).")
        small = ar[ar["bytes"] == ar["bytes"].min()]
        if not small.empty:
            best = small.loc[small["time_mean_s_mean"].idxmin()]
            out.append(f"Lowest latency allreduce ({int(best['bytes'])} B): **{best['algo']}** "
                       f"({best['time_mean_s_mean'] * 1e6:.1f} µs).")
    comp = a.get("compute_summary")
    if not comp.empty and "op" in comp.columns:
        conv = comp[comp["op"] == "conv"]
        if not conv.empty:
            gpu = conv[conv["impl"].astype(str).str.startswith(("cuda", "cublas"))]
            cpu = conv[~conv.index.isin(gpu.index)]
            if not gpu.empty and not cpu.empty:
                b = conv["batch"].max()
                tg = gpu[gpu["batch"] == b]["time_mean_s_mean"].min()
                tc = cpu[cpu["batch"] == b]["time_mean_s_mean"].min()
                if tg and tc and tg > 0:
                    out.append(f"Convolution layer at batch {int(b)}: GPU is **{tc / tg:.1f}×** faster than the "
                               f"best CPU implementation (paper reports ≈40× for a GRID K520 vs Xeon).")
    return out


def aggregate_label(row: pd.Series, cols: Sequence[str]) -> str:
    return ", ".join(f"{c}={fmt_value(row[c])}" for c in cols if c in row.index) or "default configuration"


def time_to_accuracy(a: aggregate.Analysis, target: Optional[float] = None) -> Optional[pd.DataFrame]:
    ep = a.get("epochs_summary")
    if ep.empty or "test_acc_mean" not in ep.columns or ep["test_acc_mean"].notna().sum() == 0:
        return None
    gc = [c for c in a.group_cols if c in ep.columns]
    if target is None:
        best_per = ep.groupby(gc)["test_acc_mean"].max() if gc else pd.Series([ep["test_acc_mean"].max()])
        target = float(best_per.min()) * 0.99
    rows = []
    groups = ep.groupby(gc) if gc else [((), ep)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        hit = g[g["test_acc_mean"] >= target].sort_values("epoch")
        if hit.empty or "elapsed_s_mean" not in hit.columns:
            continue
        rows.append({"config": ", ".join(f"{c}={fmt_value(v)}" for c, v in zip(gc, key)) or "all", "target": target,
                     "epoch": int(hit.iloc[0]["epoch"]), "time_to_target_s": float(hit.iloc[0]["elapsed_s_mean"])})
    return pd.DataFrame(rows)


def _section_for(a: aggregate.Analysis, report_dir: Path) -> str:
    m = a.manifest
    name = m.get("experiment", a.run_dir.parent.name)
    parts = [f"## Experiment `{name}`\n", f"Run directory: `{a.run_dir}`\n"]
    if m.get("description"):
        parts.append(f"{m['description']}\n")
    runs = m.get("runs") or []
    if runs:
        status: Dict[str, int] = {}
        for r in runs:
            status[r.get("status", "?")] = status.get(r.get("status", "?"), 0) + 1
        parts.append("Runs: " + ", ".join(f"{k}: {v}" for k, v in sorted(status.items())) + "\n")
    findings = _findings(a)
    if findings:
        parts.append("### Key findings\n\n" + "\n".join(f"- {f}" for f in findings) + "\n")

    gc = list(a.group_cols)
    s = a.get("train_summary")
    if not s.empty:
        cols = gc + ["n_runs", "train_time_s_mean", "train_time_s_ci95", "throughput_samples_per_s_mean",
                     "speedup", "efficiency", "comm_fraction_mean", "load_imbalance_mean", "final_test_acc_mean"]
        parts.append("### Training summary\n\n" + md_table(s, cols))
        phase_cols = gc + [f"phase_mean_{p}_mean" for p in aggregate.PHASES]
        parts.append("### Phase breakdown (seconds, mean over ranks)\n\n" + md_table(s, phase_cols))
    tta = time_to_accuracy(a)
    if tta is not None and not tta.empty:
        parts.append("### Time to accuracy\n\n" + md_table(tta, ["config", "target", "epoch", "time_to_target_s"]))
    comp = a.get("compute_summary")
    if not comp.empty:
        cols = ["op", "impl", "n", "batch", "time_mean_s_mean", "gflops_mean", "throughput_samples_per_s_mean",
                "rel_err_max"]
        parts.append("### Computation benchmark\n\n" + md_table(comp, cols))
    ar = a.get("comm_allreduce_summary")
    if not ar.empty:
        parts.append("### Allreduce benchmark\n\n" + md_table(
            ar, ["algo", "world_size", "bytes", "time_mean_s_mean", "time_mean_s_ci95", "busbw_Bps_mean"]))
    pm = load_json(a.run_dir / "analysis" / "perfmodel.json") if (a.run_dir / "analysis" / "perfmodel.json").exists() else None
    if pm:
        lines = ["### Performance model\n"]
        if pm.get("alpha_s") is not None:
            lines.append(f"- Hockney fit: α = {pm['alpha_s'] * 1e6:.2f} µs, 1/β = "
                         f"{(pm.get('bandwidth_Bps') or 0) / 1e9:.2f} GB/s, γ = {pm.get('gamma_s_per_byte', 0):.3g} s/B")
        if pm.get("allreduce_median_abs_rel_err") is not None:
            lines.append(f"- Allreduce model median |relative error|: {pm['allreduce_median_abs_rel_err'] * 100:.1f}%")
        if pm.get("train_median_abs_rel_err") is not None:
            lines.append(f"- Training-time model median |relative error|: {pm['train_median_abs_rel_err'] * 100:.1f}%")
        for d in pm.get("downpour") or []:
            lines.append(f"- Downpour {d['run']}: T={d['T_grad_s'] * 1e3:.2f} ms, τ={d['tau_s'] * 1e3:.3f} ms → "
                         f"k* = T/τ ≈ {d['k_star']:.1f} workers (used {d['workers']})")
        if pm.get("notes"):
            lines.append(f"- {pm['notes']}")
        parts.append("\n".join(lines) + "\n")
    checks = a.get("checks")
    if not checks.empty:
        n_fail = int((checks["passed"] == False).sum())  # noqa: E712 - pandas boolean column
        parts.append(f"### Correctness\n\n{len(checks)} checks, **{n_fail} failed**.\n\n" +
                     md_table(checks, ["name", "device", "world_size", "passed", "max_err", "tolerance", "detail"]))
    figs = sorted((a.run_dir / "figures").glob("*.png")) if (a.run_dir / "figures").is_dir() else []
    if figs:
        parts.append("### Figures\n\n" + "\n".join(f"![{f.stem}]({os.path.relpath(f, report_dir)})" for f in figs) + "\n")
    parts.append("### Environment\n\n" + _env_section(m, a.get("train")))
    return "\n".join(parts)


def build_report(run_dirs: Sequence[Path], report_dir: Path) -> str:
    sections = []
    for rd in run_dirs:
        a = aggregate.analyze(rd)
        aggregate.write(a)
        sections.append(_section_for(a, report_dir))
    head = (f"# pdl-bench report\n\nGenerated {datetime.now().isoformat(timespec='seconds')} from "
            f"{len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}. CSV and LaTeX tables are in "
            "each run's `analysis/` folder, figures (PNG + PDF) in `figures/`.\n")
    return head + "\n" + "\n---\n\n".join(sections)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+", help="run directories (or results/<experiment> for latest)")
    p.add_argument("--out", help="output Markdown file (default: <run_dir>/report.md for a single run)")


def main(args: argparse.Namespace) -> int:
    dirs = resolve_run_dirs(args.run_dirs)
    if not dirs:
        return 1
    if args.out:
        out = Path(args.out)
    else:
        out = dirs[0] / "report.md" if len(dirs) == 1 else Path("results") / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(dirs, out.parent.resolve()), encoding="utf-8")
    log(f"report -> {out}")
    return 0
