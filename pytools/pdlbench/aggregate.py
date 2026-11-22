"""Loading, flattening and statistical aggregation of result files.

`analyze(run_dir)` returns an :class:`Analysis` holding pandas DataFrames:

    train            one row per pdl_train run (config + summary + model + host)
    train_summary    grouped statistics (mean/std/min/max/count/ci95) + speedup/efficiency
    epochs           long table: one row per (run, epoch)
    epochs_summary   epochs aggregated over repetitions
    compute / compute_summary          pdl_bench_compute records
    comm_<kind> / comm_<kind>_summary  pdl_bench_comm allreduce | pingpong | bcast | alltoall
    checks           pdl_check test results

`pdlbench aggregate` writes all of them as CSV (+ LaTeX for summaries) into
`<run_dir>/analysis/`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .util import ci95, find_raw_jsons, get, load_json, log, resolve_run_dirs

PHASES = ["data", "h2d", "forward", "backward", "transfer", "sync_wait", "comm", "update", "eval", "shuffle"]
IGNORED_PARAMS = {"seed", "repeat", "run_id", "output", "tag", "log_steps", "dump_params", "verbose"}
TRAIN_METRICS = [
    "train_time_s", "wall_time_s", "throughput_samples_per_s", "samples_processed",
    "final_train_loss", "final_train_acc", "final_test_loss", "final_test_acc", "best_test_acc",
    "compute_time_mean_s", "comm_time_mean_s", "sync_wait_mean_s", "comm_fraction", "load_imbalance",
    "comm_bytes_per_rank", "comm_calls_per_rank", "time_per_step_s",
    "server_updates", "server_mean_staleness", "server_max_staleness",
] + [f"phase_mean_{p}" for p in PHASES] + [f"phase_max_{p}" for p in PHASES]
COMPUTE_KEYS = ["op", "impl", "device", "n", "M", "N", "K", "batch", "conv", "model", "include_transfer", "threads"]
COMPUTE_METRICS = ["time_mean_s", "time_min_s", "time_median_s", "gflops", "throughput_samples_per_s",
                   "max_abs_err", "rel_err"]
COMM_KEYS = {"allreduce": ["algo", "bytes", "world_size", "device"],
             "pingpong": ["bytes", "world_size", "device"],
             "bcast": ["bytes", "world_size", "device"],
             "alltoall": ["bytes", "world_size", "device"]}
COMM_METRICS = ["time_mean_s", "time_min_s", "time_max_s", "time_std_s", "algbw_Bps", "busbw_Bps",
                "one_way_s", "bandwidth_Bps", "bytes_sent_per_rank", "messages_per_rank"]


def norm(key: str) -> str:
    """Canonical column name: CLI dashes -> underscores (env.X keys are kept)."""
    return key if key.startswith("env.") else key.replace("-", "_")


@dataclass
class Analysis:
    run_dir: Path
    manifest: Dict[str, Any]
    analysis_cfg: Dict[str, Any]
    group_cols: List[str]
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame:
        return self.frames.get(name, pd.DataFrame())

    @property
    def scaling(self) -> str:
        return str(self.analysis_cfg.get("scaling", "strong"))


# ---------------------------------------------------------------------- loading

def _run_meta(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Maps result-file stem -> manifest entry."""
    out = {}
    for e in manifest.get("runs", []) or []:
        out[Path(e.get("output", e.get("name", ""))).stem] = e
    return out


def _base_row(doc: Dict[str, Any], path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run": path.stem, "repeat": meta.get("repeat", 0)}
    for k, v in (doc.get("config") or {}).items():
        if not isinstance(v, (dict, list)):
            row[norm(k)] = v
    for k, v in (meta.get("params") or {}).items():
        row.setdefault(norm(k), v if not isinstance(v, (list, dict)) else str(v))
    for k, v in (meta.get("env") or {}).items():
        row[f"env.{k}"] = v
    env = doc.get("environment") or {}
    for k in ("hostname", "cpu_model", "gpu_name", "mpi_library", "compiler", "git_commit", "impl"):
        if k in env:
            row[f"host_{k}"] = env[k]
    ws = doc.get("world_size", env.get("mpi_world_size", meta.get("np", 1)))
    row["np"] = int(ws) if ws is not None else int(meta.get("np", 1))
    row["world_size"] = row["np"]
    return row


def _train_rows(doc: Dict[str, Any], path: Path, meta: Dict[str, Any]):
    row = _base_row(doc, path, meta)
    s = doc.get("summary") or {}
    row["strategy"] = doc.get("strategy", row.get("strategy"))
    row["num_workers"] = doc.get("num_workers", row["np"])
    row["eff_local_batch"] = doc.get("local_batch")
    row["eff_global_batch"] = doc.get("global_batch")
    row["steps_per_epoch"] = doc.get("steps_per_epoch")
    row["num_params"] = get(doc, "model", "num_params")
    row["param_bytes"] = get(doc, "model", "param_bytes")
    row["flops_per_sample"] = get(doc, "model", "flops_per_sample")
    row["n_train"] = get(doc, "dataset", "n_train")
    row["partition_eff"] = get(doc, "dataset", "partition")
    for k, v in s.items():
        if isinstance(v, dict) or k == "server":
            continue
        row[k] = v
    for stat in ("mean", "max", "min"):
        for p in PHASES:
            row[f"phase_{stat}_{p}"] = get(s, f"phases_{stat}", p)
    server = s.get("server") or {}
    row["server_updates"] = server.get("updates")
    row["server_mean_staleness"] = server.get("mean_staleness")
    row["server_max_staleness"] = server.get("max_staleness")
    for p in PHASES:
        row[f"server_phase_{p}"] = get(server, "phases", p)
    epochs = doc.get("epochs") or []
    steps = sum((e.get("steps") or 0) for e in epochs)
    if steps and row.get("train_time_s"):
        row["time_per_step_s"] = row["train_time_s"] / steps
    row["total_steps"] = steps
    ep_rows = []
    for e in epochs:
        er = {k: row.get(k) for k in row if not k.startswith(("phase_", "server_"))}
        er.update({k: e.get(k) for k in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                                         "time_s", "elapsed_s", "samples", "steps")})
        for p in PHASES:
            er[f"epoch_phase_mean_{p}"] = get(e, "phases_mean", p)
        ep_rows.append(er)
    return row, ep_rows


def load(run_dir: Path) -> Analysis:
    manifest = load_json(run_dir / "manifest.json") or {}
    cfg = manifest.get("config") or {}
    analysis_cfg = cfg.get("analysis") or {}
    meta_by_stem = _run_meta(manifest)
    train, epochs, compute, checks = [], [], [], []
    param_cols: List[str] = []
    comm: Dict[str, List[Dict[str, Any]]] = {k: [] for k in COMM_KEYS}
    for path in find_raw_jsons(run_dir):
        doc = load_json(path)
        if not doc:
            continue
        meta = meta_by_stem.get(path.stem, {})
        param_cols.extend(norm(k) for k in (doc.get("config") or {}) if norm(k) not in param_cols)
        schema = str(doc.get("schema", ""))
        if schema.startswith("pdl-bench/train"):
            row, ep = _train_rows(doc, path, meta)
            train.append(row)
            epochs.extend(ep)
        elif schema.startswith("pdl-bench/compute"):
            base = _base_row(doc, path, meta)
            for rec in doc.get("records") or []:
                compute.append({**base, **{norm(k): v for k, v in rec.items()}})
        elif schema.startswith("pdl-bench/comm"):
            base = _base_row(doc, path, meta)
            base["device"] = doc.get("device", base.get("device", "cpu"))
            for kind in COMM_KEYS:
                for rec in doc.get(kind) or []:
                    comm[kind].append({**base, **rec, "world_size": base["np"]})
            bar = doc.get("barrier") or {}
            if bar:
                comm.setdefault("barrier", []).append({**base, **{f"barrier_{k}": v for k, v in bar.items()}})
        elif schema.startswith("pdl-bench/check"):
            base = _base_row(doc, path, meta)
            for t in doc.get("tests") or []:
                checks.append({**base, **t})
        else:
            log(f"warning: {path.name}: unknown schema '{schema}'")

    group_cols = _group_columns(manifest, analysis_cfg, pd.DataFrame(train or compute or checks), param_cols)
    a = Analysis(run_dir=run_dir, manifest=manifest, analysis_cfg=analysis_cfg, group_cols=group_cols)
    a.frames["train"] = pd.DataFrame(train)
    a.frames["epochs"] = pd.DataFrame(epochs)
    a.frames["compute"] = pd.DataFrame(compute)
    a.frames["checks"] = pd.DataFrame(checks)
    for kind, rows in comm.items():
        a.frames[f"comm_{kind}"] = pd.DataFrame(rows)
    return a


def _group_columns(manifest: Dict[str, Any], analysis_cfg: Dict[str, Any], df: pd.DataFrame,
                   param_cols: Sequence[str]) -> List[str]:
    """Columns identifying a configuration: analysis.group_by, else the varying run parameters."""
    if analysis_cfg.get("group_by"):
        return [norm(c) for c in analysis_cfg["group_by"]]
    cols: List[str] = []
    for e in manifest.get("runs", []) or []:
        for k in (e.get("varying") or {}):
            c = norm(k)
            if c not in cols and c not in IGNORED_PARAMS:
                cols.append(c)
    if not cols and not df.empty:
        for c in ["np", *param_cols]:
            if c in df.columns and c not in IGNORED_PARAMS and c not in cols \
                    and df[c].astype(str).nunique() > 1:
                cols.append(c)
    return cols


# ------------------------------------------------------------------ statistics

def summarize(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Groups by `keys` and computes mean/std/min/max/count/ci95 for `metrics`."""
    if df.empty:
        return pd.DataFrame()
    keys = [k for k in keys if k in df.columns]
    metrics = [m for m in metrics if m in df.columns and m not in keys]
    work = df.copy()
    for k in keys:
        work[k] = work[k].where(work[k].notna(), "-")
    for m in metrics:
        work[m] = pd.to_numeric(work[m], errors="coerce")
    if not keys:
        work["_all"] = "all"
        keys = ["_all"]
    rows = []
    for gkey, g in work.groupby(keys, sort=True):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        row: Dict[str, Any] = dict(zip(keys, gkey))
        row["n_runs"] = len(g)
        for m in metrics:
            vals = g[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            std = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = std
            row[f"{m}_min"] = float(vals.min())
            row[f"{m}_max"] = float(vals.max())
            row[f"{m}_ci95"] = ci95(std, int(vals.size))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.drop(columns=["_all"], errors="ignore")


def _match(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def add_speedup(summary: pd.DataFrame, analysis_cfg: Dict[str, Any], raw_config: Dict[str, Any],
                group_cols: Sequence[str]) -> pd.DataFrame:
    """Adds speedup / efficiency columns relative to the configured baseline rows."""
    if summary.empty or "train_time_s_mean" not in summary.columns:
        return summary
    baseline = {norm(k): v for k, v in (analysis_cfg.get("baseline") or {"np": 1}).items()}
    base_mask = pd.Series(True, index=summary.index)
    for k, v in baseline.items():
        if k not in summary.columns:
            log(f"warning: baseline key '{k}' not in results; speedup skipped")
            return summary
        base_mask &= summary[k].map(lambda x, v=v: _match(x, v))
    bases = summary[base_mask]
    if bases.empty:
        log(f"warning: no baseline rows for {baseline}; speedup skipped")
        return summary
    exprs = {norm(k) for k, v in {**(raw_config.get("base_args") or {}), **(raw_config.get("matrix") or {})}.items()
             if isinstance(v, str) and v.startswith("=")}
    if analysis_cfg.get("match_on") is not None:  # an explicit empty list means "match nothing"
        match_on = [norm(c) for c in analysis_cfg["match_on"]]
    else:
        match_on = [c for c in group_cols if c not in baseline and c not in ("np", "num_workers")
                    and c not in exprs and not c.startswith("env.")]
    match_on = [c for c in match_on if c in summary.columns]
    weak = str(analysis_cfg.get("scaling", "strong")) == "weak"
    sp, eff, tsp, bt = [], [], [], []
    for _, row in summary.iterrows():
        cand = bases
        for c in match_on:
            if cand.empty:
                break
            # .loc with a boolean mask: indexing with an empty object Series would select columns
            cand = cand.loc[cand[c].map(lambda x, v=row[c]: _match(x, v)).astype(bool)]
        if cand.empty and len(bases) == 1:
            cand = bases
        if cand.empty:
            sp.append(np.nan), eff.append(np.nan), tsp.append(np.nan), bt.append(np.nan)
            continue
        b = cand.iloc[0]
        t_b, t_r = b["train_time_s_mean"], row["train_time_s_mean"]
        s = t_b / t_r if t_r and t_r > 0 else np.nan
        procs = float(row.get("np", 1) or 1) / float(b.get("np", 1) or 1)
        thr_b = b.get("throughput_samples_per_s_mean", np.nan)
        thr_r = row.get("throughput_samples_per_s_mean", np.nan)
        ts = thr_r / thr_b if thr_b and thr_b > 0 else np.nan
        sp.append(s)
        tsp.append(ts)
        eff.append(s if weak else (s / procs if procs > 0 else np.nan))
        bt.append(t_b)
    out = summary.copy()
    out["baseline_time_s"] = bt
    out["speedup"] = sp
    out["throughput_speedup"] = tsp
    out["efficiency"] = eff
    return out


# ------------------------------------------------------------------- pipeline

def analyze(run_dir: Path) -> Analysis:
    a = load(run_dir)
    gc = a.group_cols
    train = a.get("train")
    if not train.empty:
        keys = gc or [c for c in ("strategy", "np") if c in train.columns]
        a.group_cols = keys
        s = summarize(train, keys, TRAIN_METRICS + ["num_params", "param_bytes", "eff_local_batch",
                                                    "eff_global_batch", "steps_per_epoch", "total_steps"])
        a.frames["train_summary"] = add_speedup(s, a.analysis_cfg, a.manifest.get("config") or {}, keys)
    ep = a.get("epochs")
    if not ep.empty:
        a.frames["epochs_summary"] = summarize(ep, list(a.group_cols) + ["epoch"],
                                               ["train_loss", "train_acc", "test_loss", "test_acc",
                                                "time_s", "elapsed_s"])
    comp = a.get("compute")
    if not comp.empty:
        a.frames["compute_summary"] = summarize(comp, COMPUTE_KEYS + [c for c in gc if c not in COMPUTE_KEYS],
                                                COMPUTE_METRICS)
    for kind, keys in COMM_KEYS.items():
        df = a.get(f"comm_{kind}")
        if not df.empty:
            a.frames[f"comm_{kind}_summary"] = summarize(df, keys, COMM_METRICS)
    return a


def _latex(df: pd.DataFrame, path: Path, max_cols: int = 12) -> None:
    cols = [c for c in df.columns if not c.endswith(("_min", "_max", "_std"))][:max_cols]
    sub = df[cols]
    try:
        text = sub.to_latex(index=False, float_format=lambda x: f"{x:.4g}")
    except (ImportError, AttributeError, ValueError):
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.4g}"
            return str(v).replace("_", r"\_").replace("%", r"\%")
        lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
                 " & ".join(fmt(c) for c in cols) + r" \\", r"\hline"]
        for _, r in sub.iterrows():
            lines.append(" & ".join(fmt(r[c]) for c in cols) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def write(a: Analysis) -> Path:
    out = a.run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in a.frames.items():
        if df is None or df.empty:
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        if name.endswith("_summary"):
            _latex(df, out / f"{name}.tex")
    return out


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+", help="run directories (or results/<experiment> for latest)")


def main(args: argparse.Namespace) -> int:
    for run_dir in resolve_run_dirs(args.run_dirs):
        a = analyze(run_dir)
        out = write(a)
        sizes = {k: len(v) for k, v in a.frames.items() if v is not None and not v.empty}
        log(f"{run_dir}: {sizes} -> {out}")
    return 0
