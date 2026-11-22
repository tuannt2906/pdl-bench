"""Figures for research reports (PNG at 200 dpi + vector PDF).

Every figure function takes an :class:`~pdlbench.aggregate.Analysis` and the output
directory, and returns the list of written files (empty when the required data is
absent). `PLOTS` maps figure names (usable in `analysis.plots`) to functions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

from . import aggregate  # noqa: E402
from .util import fmt_value, human_bytes, log, resolve_run_dirs  # noqa: E402

PHASE_GROUPS = {
    "compute": ["data", "h2d", "forward", "backward"],
    "update": ["update"],
    "transfer (D2H/H2D)": ["transfer"],
    "sync wait": ["sync_wait"],
    "communication": ["comm"],
    "shuffle": ["shuffle"],
    "evaluation": ["eval"],
}
GPU_IMPLS = ("cuda", "cublas")


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.2),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def save(fig: plt.Figure, out: Path, name: str) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
        files.append(p)
    plt.close(fig)
    return files


# --------------------------------------------------------------------- helpers

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hue_cols(df: pd.DataFrame, group_cols: Sequence[str], exclude: Sequence[str]) -> List[str]:
    return [c for c in group_cols if c in df.columns and c not in exclude and df[c].astype(str).nunique() > 1]


def _label(row: pd.Series, cols: Sequence[str]) -> str:
    return ", ".join(f"{c.replace('env.', '')}={fmt_value(row[c])}" for c in cols) if cols else "all"


def _lines(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, hue: Sequence[str], err: Optional[str] = None,
           marker: str = "o") -> None:
    groups = df.groupby(list(hue), sort=True) if hue else [((), df)]
    for key, g in groups:
        g = g.copy()
        g["_x"] = _num(g[x])
        g = g.dropna(subset=["_x"]).sort_values("_x")
        yv = _num(g[y])
        if yv.notna().sum() == 0:
            continue
        label = _label(g.iloc[0], hue)
        if err and err.endswith("_ci95") and "n_runs" in g.columns and _num(g["n_runs"]).min() < 3:
            # a t-based 95% CI from 2 runs (t = 12.7) is uninformative: show the std instead
            err = err[: -len("_ci95")] + "_std"
        if err and err in g.columns and _num(g[err]).notna().any():
            ax.errorbar(g["_x"], yv, yerr=_num(g[err]).fillna(0), marker=marker, capsize=3, label=label)
        else:
            ax.plot(g["_x"], yv, marker=marker, label=label)


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()


def _x_col(a: aggregate.Analysis, df: pd.DataFrame) -> Optional[str]:
    """Preferred x axis (analysis.x, else np), falling back to the first numeric group column."""
    cands = [aggregate.norm(str(a.analysis_cfg.get("x", "np"))), "np", *a.group_cols]
    for c in cands:
        if c in df.columns and _num(df[c]).nunique() > 1:
            return c
    return None


def _train_summary(a: aggregate.Analysis) -> pd.DataFrame:
    return a.get("train_summary")


# ---------------------------------------------------------------- training plots

def _simple_train(a: aggregate.Analysis, out: Path, name: str, y: str, ylabel: str, title: str,
                  ideal: Optional[str] = None) -> List[Path]:
    df = _train_summary(a)
    if df.empty or y not in df.columns:
        return []
    x = _x_col(a, df)
    if x is None:
        return []
    hue = _hue_cols(df, a.group_cols, [x])
    fig, ax = plt.subplots()
    err = y.replace("_mean", "_ci95") if y.endswith("_mean") else None
    _lines(ax, df, x, y, hue, err)
    xs = np.sort(_num(df[x]).dropna().unique())
    if ideal == "linear" and xs.size:
        ax.plot(xs, xs / xs[0], "k--", lw=1, label="ideal")
    elif ideal == "one":
        ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
    _finish(ax, title, x, ylabel)
    return save(fig, out, name)


def plot_training_time(a, out):
    return _simple_train(a, out, "training_time", "train_time_s_mean", "training time [s]",
                         "Training time (mean ± 95% CI)")


def plot_speedup(a, out):
    return _simple_train(a, out, "speedup", "speedup", "speedup T(baseline)/T", "Speedup", ideal="linear")


def plot_efficiency(a, out):
    return _simple_train(a, out, "efficiency", "efficiency", "parallel efficiency", "Parallel efficiency",
                         ideal="one")


def plot_throughput(a, out):
    return _simple_train(a, out, "throughput", "throughput_samples_per_s_mean", "samples / s",
                         "Training throughput")


def plot_phase_breakdown(a, out):
    df = _train_summary(a)
    cols = {g: [f"phase_mean_{p}_mean" for p in ps if f"phase_mean_{p}_mean" in df.columns]
            for g, ps in PHASE_GROUPS.items()}
    if df.empty or not any(cols.values()):
        return []
    labels = [_label(r, a.group_cols) for _, r in df.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 0.55 * len(df) + 4), 4.4))
    for ax, normalize in zip(axes, (False, True)):
        bottom = np.zeros(len(df))
        totals = np.zeros(len(df))
        vals = {}
        for g, cs in cols.items():
            if not cs:
                continue
            v = df[cs].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).to_numpy()
            if not v.any():
                continue
            vals[g] = v
            totals += v
        for g, v in vals.items():
            h = v / np.where(totals > 0, totals, 1) * 100 if normalize else v
            ax.bar(range(len(df)), h, bottom=bottom, label=g)
            bottom += h
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        _finish(ax, "Time share per phase" if normalize else "Time per phase (mean over ranks)",
                "", "% of time" if normalize else "seconds", legend=not normalize)
    fig.tight_layout()
    return save(fig, out, "phase_breakdown")


def plot_strong_scaling(a, out):
    df = _train_summary(a)
    if df.empty or "np" not in df.columns or _num(df["np"]).nunique() < 2:
        return []
    hue = _hue_cols(df, a.group_cols, ["np"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines(ax1, df, "np", "train_time_s_mean", hue, "train_time_s_ci95")
    base = df.sort_values("np").iloc[0]
    xs = np.sort(_num(df["np"]).dropna().unique())
    ax1.plot(xs, base["train_time_s_mean"] * xs[0] / xs, "k--", lw=1, label="ideal T1/p")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    _finish(ax1, "Strong scaling: time", "processes", "training time [s]")
    if "efficiency" in df.columns:
        _lines(ax2, df, "np", "efficiency", hue)
        ax2.axhline(1, color="k", ls="--", lw=1)
        ax2.set_ylim(0, max(1.1, float(_num(df["efficiency"]).max() or 1) * 1.1))
    _finish(ax2, "Strong scaling: efficiency", "processes", "S(p)/p")
    fig.tight_layout()
    return save(fig, out, "strong_scaling")


def plot_weak_scaling(a, out):
    df = _train_summary(a)
    if df.empty or "np" not in df.columns or _num(df["np"]).nunique() < 2:
        return []
    hue = _hue_cols(df, a.group_cols, ["np"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    work = df.copy()
    if "baseline_time_s" in work.columns:
        work["weak_eff"] = _num(work["baseline_time_s"]) / _num(work["train_time_s_mean"])
        _lines(ax1, work, "np", "weak_eff", hue)
        ax1.axhline(1, color="k", ls="--", lw=1)
    _finish(ax1, "Weak scaling efficiency T1/Tp", "processes", "efficiency")
    _lines(ax2, df, "np", "throughput_samples_per_s_mean", hue, "throughput_samples_per_s_ci95")
    _finish(ax2, "Weak scaling throughput", "processes", "samples / s")
    fig.tight_layout()
    return save(fig, out, "weak_scaling")


def plot_convergence(a, out):
    df = a.get("epochs_summary")
    if df.empty or "epoch" not in df.columns:
        return []
    metric = "test_acc_mean" if "test_acc_mean" in df.columns and df["test_acc_mean"].notna().any() \
        else "train_loss_mean"
    hue = _hue_cols(df, a.group_cols, ["epoch"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines(ax1, df.dropna(subset=[metric]), "epoch", metric, hue, metric.replace("_mean", "_ci95"))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    _finish(ax1, "Convergence per epoch", "epoch", metric.replace("_mean", ""))
    if "elapsed_s_mean" in df.columns:
        _lines(ax2, df.dropna(subset=[metric]), "elapsed_s_mean", metric, hue)
    _finish(ax2, "Convergence vs training time", "training time [s]", metric.replace("_mean", ""),
            legend=False)
    fig.tight_layout()
    return save(fig, out, "convergence")


def plot_model_size(a, out):
    df = _train_summary(a)
    if df.empty or "num_params_mean" not in df.columns or _num(df["num_params_mean"]).nunique() < 2:
        return []
    hue = _hue_cols(df, a.group_cols, ["width", "model", "num_params"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    y1 = "time_per_step_s_mean" if "time_per_step_s_mean" in df.columns else "train_time_s_mean"
    _lines(ax1, df, "num_params_mean", y1, hue)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    _finish(ax1, "Model size sensitivity", "parameters", y1.replace("_mean", ""))
    if "comm_fraction_mean" in df.columns:
        _lines(ax2, df, "num_params_mean", "comm_fraction_mean", hue)
        ax2.set_xscale("log")
    _finish(ax2, "Communication fraction", "parameters", "(comm+transfer+wait)/time", legend=False)
    fig.tight_layout()
    return save(fig, out, "model_size")


def plot_batch_size(a, out):
    files: List[Path] = []
    df = _train_summary(a)
    x = next((c for c in ("batch_size", "global_batch") if c in a.group_cols and c in df.columns
              and _num(df[c]).nunique() > 1), None)
    if not df.empty and x:
        hue = _hue_cols(df, a.group_cols, [x, "batch_size", "global_batch", "eff_local_batch"])
        fig, ax = plt.subplots()
        _lines(ax, df, x, "throughput_samples_per_s_mean", hue, "throughput_samples_per_s_ci95")
        ax.set_xscale("log", base=2)
        _finish(ax, "Batch size sensitivity (training)", "batch size per rank", "samples / s")
        files += save(fig, out, "batch_size")
    comp = a.get("compute_summary")
    if not comp.empty and "op" in comp.columns:
        ts = comp[comp["op"] == "train-step"]
        if not ts.empty and "throughput_samples_per_s_mean" in ts.columns:
            hue = [c for c in ("impl", "model") if c in ts.columns and ts[c].astype(str).nunique() > 1]
            fig, ax = plt.subplots()
            _lines(ax, ts, "batch", "throughput_samples_per_s_mean", hue or ["impl"])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _finish(ax, "Training step throughput vs batch", "batch size", "samples / s")
            files += save(fig, out, "batch_size_train_step")
    return files


# ---------------------------------------------------------------- compute plots

def _gpu_ratio(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """CPU time / GPU time for each GPU impl, against the best CPU impl at each x."""
    cpu = df[~df["impl"].astype(str).str.startswith(GPU_IMPLS)]
    gpu = df[df["impl"].astype(str).str.startswith(GPU_IMPLS)]
    if cpu.empty or gpu.empty:
        return pd.DataFrame()
    best_cpu = cpu.groupby(x)["time_mean_s_mean"].min()
    rows = []
    for _, r in gpu.iterrows():
        if r[x] in best_cpu.index and r["time_mean_s_mean"] > 0:
            rows.append({x: r[x], "impl": r["impl"], "ratio": best_cpu[r[x]] / r["time_mean_s_mean"]})
    return pd.DataFrame(rows)


def plot_compute_cpu_gpu(a, out):
    df = a.get("compute_summary")
    if df.empty or "op" not in df.columns:
        return []
    files: List[Path] = []
    for op, x, xlabel, tag in (("conv", "batch", "batch size", "conv"), ("fc", "n", "matrix size n", "fc")):
        d = df[df["op"] == op]
        if d.empty or x not in d.columns or _num(d[x]).nunique() < 2:
            continue
        if op == "fc" and "batch" in d.columns and d["batch"].astype(str).nunique() > 1:
            d = d[_num(d["batch"]) == _num(d["batch"]).min()]
        ratio = _gpu_ratio(d, x)
        if ratio.empty:  # CPU-only run: no GPU speedup panel
            fig, ax1 = plt.subplots(figsize=(6, 4.2))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        _lines(ax1, d, x, "time_mean_s_mean", ["impl"])
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        _finish(ax1, f"{op.upper()} forward time (paper Fig. {'1' if op == 'conv' else '3'})", xlabel, "time [s]")
        if not ratio.empty:
            _lines(ax2, ratio, x, "ratio", ["impl"])
            ax2.set_xscale("log", base=2)
            _finish(ax2, f"GPU speedup over best CPU (paper Fig. {'2' if op == 'conv' else '4'})", xlabel,
                    "CPU time / GPU time")
        fig.tight_layout()
        files += save(fig, out, f"compute_{tag}_cpu_gpu")
    return files


def plot_gemm(a, out):
    df = a.get("compute_summary")
    if df.empty or "op" not in df.columns:
        return []
    d = df[df["op"] == "gemm"]
    if d.empty or "gflops_mean" not in d.columns:
        return []
    fig, ax = plt.subplots()
    _lines(ax, d, "n", "gflops_mean", ["impl"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, "GEMM performance (N×N×N)", "N", "GFLOP/s")
    return save(fig, out, "gemm")


# ------------------------------------------------------------------- comm plots

def plot_comm_allreduce(a, out):
    df = a.get("comm_allreduce_summary")
    if df.empty:
        return []
    files: List[Path] = []
    for ws, d in df.groupby("world_size"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        _lines(ax1, d, "bytes", "time_mean_s_mean", ["algo"])
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        _finish(ax1, f"Allreduce time (p={ws})", "message size [bytes]", "time [s]")
        if "busbw_Bps_mean" in d.columns:
            dd = d.copy()
            dd["busbw_GBps"] = _num(dd["busbw_Bps_mean"]) / 1e9
            _lines(ax2, dd, "bytes", "busbw_GBps", ["algo"])
            ax2.set_xscale("log", base=2)
        _finish(ax2, f"Allreduce bus bandwidth (p={ws})", "message size [bytes]", "GB/s", legend=False)
        fig.tight_layout()
        files += save(fig, out, f"comm_allreduce_p{ws}")
    return files


def plot_comm_scaling(a, out):
    df = a.get("comm_allreduce_summary")
    if df.empty or _num(df["world_size"]).nunique() < 2:
        return []
    sizes = np.sort(_num(df["bytes"]).dropna().unique())
    picks = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.0), squeeze=False)
    for ax, size in zip(axes[0], picks):
        d = df[_num(df["bytes"]) == size]
        _lines(ax, d, "world_size", "time_mean_s_mean", ["algo"])
        ax.set_yscale("log")
        _finish(ax, f"Allreduce {human_bytes(size)}", "processes", "time [s]", legend=ax is axes[0][0])
    fig.tight_layout()
    return save(fig, out, "comm_scaling")


def plot_pingpong(a, out):
    df = a.get("comm_pingpong_summary")
    if df.empty or "one_way_s_mean" not in df.columns:
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines(ax1, df, "bytes", "one_way_s_mean", [])
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    _finish(ax1, "Ping-pong one-way latency", "message size [bytes]", "time [s]", legend=False)
    if "bandwidth_Bps_mean" in df.columns:
        d = df.copy()
        d["GBps"] = _num(d["bandwidth_Bps_mean"]) / 1e9
        _lines(ax2, d, "bytes", "GBps", [])
        ax2.set_xscale("log", base=2)
    _finish(ax2, "Ping-pong bandwidth", "message size [bytes]", "GB/s", legend=False)
    fig.tight_layout()
    return save(fig, out, "pingpong")


def plot_perfmodel(a, out):
    files: List[Path] = []
    path = a.run_dir / "analysis" / "perfmodel.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            fig, ax = plt.subplots()
            for algo, g in df.groupby("algo"):
                ax.scatter(g["measured_s"], g["predicted_s"], s=14, label=algo)
            lo = float(np.nanmin(df[["measured_s", "predicted_s"]].to_numpy()))
            hi = float(np.nanmax(df[["measured_s", "predicted_s"]].to_numpy()))
            if math.isfinite(lo) and lo > 0:
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _finish(ax, "Allreduce: α-β model vs measured", "measured [s]", "predicted [s]")
            files += save(fig, out, "perfmodel_allreduce")
    path = a.run_dir / "analysis" / "perfmodel_train.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty and "np" in df.columns:
            fig, ax = plt.subplots()
            for grp, g in df.groupby("group", sort=True):
                g = g.sort_values("np")
                ax.plot(g["np"], g["measured_speedup"], "o-", label=f"{grp} measured")
                ax.plot(g["np"], g["predicted_speedup"], "x--", label=f"{grp} model")
            _finish(ax, "Training speedup: model vs measured", "processes", "speedup")
            files += save(fig, out, "perfmodel_train")
    return files


PLOTS: Dict[str, Callable[[aggregate.Analysis, Path], List[Path]]] = {
    "training_time": plot_training_time,
    "speedup": plot_speedup,
    "efficiency": plot_efficiency,
    "throughput": plot_throughput,
    "phase_breakdown": plot_phase_breakdown,
    "strong_scaling": plot_strong_scaling,
    "weak_scaling": plot_weak_scaling,
    "convergence": plot_convergence,
    "model_size": plot_model_size,
    "batch_size": plot_batch_size,
    "compute_cpu_gpu": plot_compute_cpu_gpu,
    "gemm": plot_gemm,
    "comm_allreduce": plot_comm_allreduce,
    "comm_scaling": plot_comm_scaling,
    "pingpong": plot_pingpong,
    "perfmodel": plot_perfmodel,
}


def plot_all(a: aggregate.Analysis, names: Optional[Sequence[str]] = None) -> List[Path]:
    apply_style()
    out = a.run_dir / "figures"
    wanted = list(names or a.analysis_cfg.get("plots") or PLOTS.keys())
    if not names and not a.analysis_cfg.get("plots"):
        wanted.remove("strong_scaling" if a.scaling == "weak" else "weak_scaling")
    files: List[Path] = []
    for name in wanted:
        fn = PLOTS.get(name)
        if fn is None:
            log(f"warning: unknown plot '{name}' (choices: {', '.join(PLOTS)})")
            continue
        try:
            files += fn(a, out)
        except (KeyError, ValueError, TypeError) as exc:
            log(f"warning: plot '{name}' failed: {exc}")
    return files


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+", help="run directories (or results/<experiment> for latest)")
    p.add_argument("--only", nargs="+", metavar="PLOT", help=f"subset of figures: {', '.join(PLOTS)}")


def main(args: argparse.Namespace) -> int:
    for run_dir in resolve_run_dirs(args.run_dirs):
        a = aggregate.analyze(run_dir)
        aggregate.write(a)
        files = plot_all(a, args.only)
        log(f"{run_dir}: {len(files) // 2} figures -> {run_dir / 'figures'}")
    return 0
