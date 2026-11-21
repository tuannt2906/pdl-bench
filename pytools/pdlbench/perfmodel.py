"""Analytical performance models compared against measurements.

Communication (Hockney α-β model, paper §5 uses latency L and bandwidth B):
    T_p2p(m) = α + β·m                     fitted from the ping-pong benchmark
Allreduce of n bytes over k processes (γ = local reduction cost per byte, optional):
    ring          2(k−1)·α + 2(k−1)/k·n·β + (k−1)/k·n·γ
    rd            ⌈log2 k⌉·(α + n·β) + ⌈log2 k⌉·n·γ      (+2 extra steps if k is not a power of 2)
    tree          2⌈log2 k⌉·(α + n·β) + ⌈log2 k⌉·n·γ      ("BitTorrent aggregate" + broadcast)
    naive         2(k−1)·(α + n·β) + (k−1)·n·γ           (central driver / parameter server)
    reduce_bcast  same as tree
    mpi           min(ring, rd, tree)                     (libraries switch algorithms by size)

Training (synchronous SGD, fused gradients of P bytes, S steps per worker):
    T(k) = T_comp(k) + S·T_allreduce(P, k)
    T_comp(k) = T_comp(k0)·k0/k (strong scaling) or T_comp(k0) (weak scaling)

Downpour SGD (paper §5.2.1): with gradient time T per push and server update time τ,
adding workers stops helping beyond k* = T/τ.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import aggregate
from .util import dump_json, log, resolve_run_dirs

ALGOS = ("ring", "rd", "tree", "naive", "reduce_bcast", "mpi")


def fit_alpha_beta(pingpong: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """Fits T = α + β·m minimising the *relative* error (small and large messages count equally)."""
    if pingpong.empty or "one_way_s_mean" not in pingpong.columns:
        return None
    d = pingpong[["bytes", "one_way_s_mean"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[d["one_way_s_mean"] > 0]
    if len(d) < 2:
        return None
    m, t = d["bytes"].to_numpy(float), d["one_way_s_mean"].to_numpy(float)
    A = np.stack([1.0 / t, m / t], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.ones_like(t), rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    if beta <= 0:  # degenerate fit: fall back to the largest-message bandwidth
        beta = float(t[np.argmax(m)] / m.max())
    return max(alpha, 0.0), beta


def _log2c(k: int) -> int:
    return int(math.ceil(math.log2(k))) if k > 1 else 0


def gamma_coeff(algo: str, k: int, n: float) -> float:
    lg = _log2c(k)
    return {"ring": (k - 1) / k * n, "rd": lg * n, "tree": lg * n, "reduce_bcast": lg * n,
            "naive": (k - 1) * n}.get(algo, (k - 1) / k * n)


def predict_allreduce(algo: str, k: int, n: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    if k <= 1:
        return 0.0
    lg = _log2c(k)
    if algo == "ring":
        return 2 * (k - 1) * alpha + 2 * (k - 1) / k * n * beta + gamma * gamma_coeff(algo, k, n)
    if algo == "rd":
        extra = 0 if (k & (k - 1)) == 0 else 2
        return (lg + extra) * (alpha + n * beta) + gamma * gamma_coeff(algo, k, n)
    if algo in ("tree", "reduce_bcast"):
        return 2 * lg * (alpha + n * beta) + gamma * gamma_coeff(algo, k, n)
    if algo == "naive":
        return 2 * (k - 1) * (alpha + n * beta) + gamma * gamma_coeff(algo, k, n)
    return min(predict_allreduce(a, k, n, alpha, beta, gamma) for a in ("ring", "rd", "tree"))


def fit_gamma(allreduce: pd.DataFrame, alpha: float, beta: float, min_bytes: int = 65536) -> float:
    """Least-squares γ ≥ 0 on the residual of large-message allreduce measurements."""
    if allreduce.empty:
        return 0.0
    num = den = 0.0
    for _, r in allreduce.iterrows():
        n, k = float(r["bytes"]), int(r["world_size"])
        if n < min_bytes or k < 2 or str(r["algo"]) == "mpi":
            continue
        res = float(r["time_mean_s_mean"]) - predict_allreduce(str(r["algo"]), k, n, alpha, beta)
        f = gamma_coeff(str(r["algo"]), k, n)
        num += res * f
        den += f * f
    return max(num / den, 0.0) if den > 0 else 0.0


def comm_table(allreduce: pd.DataFrame, alpha: float, beta: float, gamma: float) -> pd.DataFrame:
    rows = []
    for _, r in allreduce.iterrows():
        algo, k, n = str(r["algo"]), int(r["world_size"]), float(r["bytes"])
        meas = float(r["time_mean_s_mean"])
        p0 = predict_allreduce(algo, k, n, alpha, beta)
        pg = predict_allreduce(algo, k, n, alpha, beta, gamma)
        rows.append({"algo": algo, "world_size": k, "bytes": int(n), "measured_s": meas,
                     "predicted_ab_s": p0, "predicted_s": pg,
                     "rel_err": (pg - meas) / meas if meas > 0 else np.nan})
    return pd.DataFrame(rows)


def train_table(a: aggregate.Analysis, alpha: float, beta: float, gamma: float) -> pd.DataFrame:
    s = a.get("train_summary")
    need = {"np", "train_time_s_mean", "compute_time_mean_s_mean", "param_bytes_mean", "total_steps_mean"}
    if s.empty or not need.issubset(s.columns):
        return pd.DataFrame()
    weak = a.scaling == "weak"
    group = [c for c in a.group_cols if c in s.columns and c != "np"]
    rows: List[Dict[str, Any]] = []
    grouped = s.groupby(group, sort=True) if group else [((), s)]
    for key, g in grouped:
        key = key if isinstance(key, tuple) else (key,)
        g = g.sort_values("np")
        base = g.iloc[0]
        k0 = int(base["np"])
        strategy = str(base.get("strategy", "sync"))
        for _, r in g.iterrows():
            k = int(r["np"])
            t_comp = float(base["compute_time_mean_s_mean"]) * (1.0 if weak else k0 / k)
            algo = str(r.get("allreduce", "mpi")) if "allreduce" in r else "mpi"
            algo = algo if algo in ALGOS else "mpi"
            calls = float(r.get("comm_calls_per_rank_mean", np.nan))
            n_ar = calls if strategy != "sync" and math.isfinite(calls) else float(r["total_steps_mean"])
            msg = float(r["param_bytes_mean"])
            if str(r.get("fusion", "flat")) == "layer" and math.isfinite(calls) and calls > 0:
                n_ar, msg = calls, msg * float(r["total_steps_mean"]) / calls
            t_comm = n_ar * predict_allreduce(algo, k, msg, alpha, beta, gamma) if k > 1 else 0.0
            pred = t_comp + t_comm
            meas = float(r["train_time_s_mean"])
            t_base = float(base["train_time_s_mean"])
            rows.append({"group": ", ".join(f"{c}={v}" for c, v in zip(group, key)) or "all", "np": k,
                         "strategy": strategy, "measured_time_s": meas, "predicted_time_s": pred,
                         "predicted_compute_s": t_comp, "predicted_comm_s": t_comm,
                         "measured_comm_s": float(r.get("comm_time_mean_s_mean", np.nan)),
                         "measured_speedup": t_base / meas if meas > 0 else np.nan,
                         "predicted_speedup": t_base / pred if pred > 0 else np.nan,
                         "rel_err": (pred - meas) / meas if meas > 0 else np.nan})
    return pd.DataFrame(rows)


def downpour_table(a: aggregate.Analysis) -> List[Dict[str, Any]]:
    t = a.get("train")
    if t.empty or "server_updates" not in t.columns:
        return []
    out = []
    for _, r in t.iterrows():
        upd = r.get("server_updates")
        steps = r.get("total_steps")
        if not upd or not steps or pd.isna(upd):
            continue
        push = float(r.get("push_every", 1) or 1)
        t_grad = float(r.get("compute_time_mean_s", np.nan)) / float(steps) * push
        tau = sum(float(r.get(f"server_phase_{p}") or 0.0) for p in ("update", "comm")) / float(upd)
        out.append({"run": r["run"], "np": int(r["np"]), "workers": int(r.get("num_workers", r["np"] - 1)),
                    "T_grad_s": t_grad, "tau_s": tau,
                    "k_star": t_grad / tau if tau > 0 else math.inf,
                    "mean_staleness": r.get("server_mean_staleness")})
    return out


def analyze(run_dir: Path, comm_dir: Optional[Path] = None) -> Dict[str, Any]:
    a = aggregate.analyze(run_dir)
    out_dir = aggregate.write(a)
    src = aggregate.analyze(comm_dir) if comm_dir else a
    pp = src.get("comm_pingpong_summary")
    fit = fit_alpha_beta(pp)
    result: Dict[str, Any] = {"run_dir": str(run_dir), "comm_source": str(src.run_dir)}
    if fit is None:
        log(f"{run_dir}: no ping-pong data (pass --comm-dir with a comm_allreduce run); "
            "communication model skipped")
    else:
        alpha, beta = fit
        ar = src.get("comm_allreduce_summary")
        gamma = fit_gamma(ar, alpha, beta)
        result.update(alpha_s=alpha, beta_s_per_byte=beta, bandwidth_Bps=1.0 / beta if beta > 0 else None,
                      gamma_s_per_byte=gamma)
        has_own = not a.get("comm_allreduce_summary").empty or not a.get("train").empty
        if not ar.empty and has_own:
            ct = comm_table(ar, alpha, beta, gamma)
            ct.to_csv(out_dir / "perfmodel.csv", index=False)
            result["allreduce_median_abs_rel_err"] = float(ct["rel_err"].abs().median())
            result["allreduce_by_algo"] = {
                algo: float(g["rel_err"].abs().median()) for algo, g in ct.groupby("algo")}
        tt = train_table(a, alpha, beta, gamma)
        if not tt.empty:
            tt.to_csv(out_dir / "perfmodel_train.csv", index=False)
            result["train_median_abs_rel_err"] = float(tt["rel_err"].abs().median())
    dp = downpour_table(a)
    if dp:
        pd.DataFrame(dp).to_csv(out_dir / "perfmodel_downpour.csv", index=False)
        result["downpour"] = dp
    result["notes"] = ("Single-node runs: MPI uses shared memory, so α/β describe intra-node copies and the "
                       "model ignores memory-bandwidth contention between ranks.")
    dump_json(result, out_dir / "perfmodel.json")
    return result


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dirs", nargs="+", help="run directories (or results/<experiment> for latest)")
    p.add_argument("--comm-dir", help="run directory of a comm benchmark providing ping-pong/allreduce data "
                                      "(default: the run directory itself)")


def main(args: argparse.Namespace) -> int:
    comm_dir = resolve_run_dirs([args.comm_dir])[0] if args.comm_dir else None
    for run_dir in resolve_run_dirs(args.run_dirs):
        res = analyze(run_dir, comm_dir)
        keys = ("alpha_s", "beta_s_per_byte", "gamma_s_per_byte", "allreduce_median_abs_rel_err",
                "train_median_abs_rel_err")
        log(f"{run_dir}: " + ", ".join(f"{k}={res[k]:.3g}" for k in keys if res.get(k) is not None))
    return 0
