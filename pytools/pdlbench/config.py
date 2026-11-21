"""Experiment configuration (YAML) and expansion into concrete runs.

Schema (all keys optional except `name`)::

    name: strong_scaling            # experiment name -> results/<name>/<timestamp>/
    description: "..."
    binary: pdl_train               # pdl_train | pdl_bench_compute | pdl_bench_comm | pdl_check
    repeats: 3                      # repetitions of every parameter combination
    vary_seed: true                 # seed = seed_base + repeat (else seed = seed_base)
    seed_base: 42
    timeout: 3600                   # seconds per run
    mpi:
      launcher: mpirun
      np: [1, 2, 4]                 # process counts (implicit matrix key `np`)
      extra_args: [--bind-to, core, --map-by, core]
      allow_oversubscribe: true     # add --oversubscribe / hwthread binding when np > cores
      launch_single: false          # np == 1 runs the binary directly (no mpirun)
    env: {OMP_NUM_THREADS: 1}       # environment of every run
    base_args: {strategy: sync, epochs: 3}   # CLI options (`--key value`)
    matrix:                         # cartesian product; `np` and `env.X` keys allowed
      allreduce: [mpi, ring]
    variants:                       # optional list of overrides crossed with the matrix
      - {strategy: sequential, np: 1}
      - {strategy: sync}
    exclude:                        # optional partial parameter sets to skip
      - {device: cuda, np: 8}
    analysis:
      group_by: [strategy, np]      # columns identifying a configuration
      baseline: {np: 1}             # reference rows for speedup / efficiency
      match_on: [strategy]          # keys a row must share with its baseline (optional)
      scaling: strong               # strong | weak
      x: np                         # preferred x axis for line plots
      plots: [speedup, efficiency]  # figures (default: auto-detected)

Values: booleans become 1/0, lists are comma-joined, `null` drops the option.
Strings starting with `=` are Python expressions evaluated with ``np``, ``repeat``,
``seed``, ``min``, ``max``, ``int``, ``float``, ``round`` and every other resolved
parameter (dashes replaced by underscores), e.g. ``train-limit: "=6000*np"``.
The keys `binary` and `np` may also be set inside `matrix` / `variants`.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

#: Binaries that accept --seed / --run-id / --tag (pdl_check only takes --output).
SEEDED_BINARIES = {"pdl_train", "pdl_bench_compute", "pdl_bench_comm"}
#: Parameter keys that are not passed through as CLI options.
RESERVED = {"np", "binary"}
_SAFE_FUNCS = {"min": min, "max": max, "int": int, "float": float, "round": round, "abs": abs}


@dataclass
class Run:
    """One concrete invocation of a benchmark binary."""

    index: int
    name: str
    binary: str
    np: int
    repeat: int
    params: Dict[str, Any]          # resolved CLI options (without np/binary/env)
    env: Dict[str, str]
    varying: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    name: str
    raw: Dict[str, Any]
    path: Optional[Path] = None

    @property
    def binary(self) -> str:
        return str(self.raw.get("binary", "pdl_train"))

    @property
    def mpi(self) -> Dict[str, Any]:
        return self.raw.get("mpi") or {}

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.raw.get("analysis") or {}

    @property
    def timeout(self) -> float:
        return float(self.raw.get("timeout", 3600))

    @property
    def description(self) -> str:
        return str(self.raw.get("description", "")).strip()


def load(path: Path) -> Experiment:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "name" not in raw:
        raw["name"] = path.stem
    return Experiment(name=str(raw["name"]), raw=raw, path=path)


def _as_list(v: Any) -> List[Any]:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _eval(expr: str, ns: Dict[str, Any]) -> Any:
    code = compile(expr, "<expr>", "eval")
    for name in code.co_names:
        if name not in ns and name not in _SAFE_FUNCS:
            raise NameError(name)
    return eval(code, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})  # noqa: S307 - restricted namespace


def _resolve(values: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates `=expr` strings; repeats until all dependencies are resolved."""
    out = dict(values)
    pending = {k for k, v in out.items() if isinstance(v, str) and v.startswith("=")}
    for _ in range(len(pending) + 1):
        if not pending:
            break
        ns = {k.replace("-", "_").replace(".", "_"): v for k, v in out.items() if k not in pending}
        ns.update(extra)
        progressed = False
        for k in sorted(pending):
            try:
                out[k] = _eval(out[k][1:], ns)
                pending.discard(k)
                progressed = True
            except NameError:
                continue
        if not progressed:
            break
    if pending:
        raise ValueError(f"unresolvable expressions: {sorted(pending)}")
    return out


def _slug(varying: Dict[str, Any], max_len: int = 72) -> str:
    parts = []
    for k, v in varying.items():
        if v is None:
            continue
        key = re.sub(r"[^A-Za-z0-9]", "", k.replace("env.", ""))[:8]
        val = str(v).replace("pdl_", "")
        val = re.sub(r"[^A-Za-z0-9.]+", "-", val).strip("-")[:16]
        parts.append(f"{key}-{val}")
    s = "_".join(parts)
    return s[:max_len] if s else "run"


def format_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return str(v)


def expand(exp: Experiment, repeats: Optional[int] = None,
           filters: Optional[List[Tuple[str, str]]] = None) -> List[Run]:
    """Expands the experiment into the ordered list of runs (deduplicated)."""
    raw = exp.raw
    base = dict(raw.get("base_args") or {})
    matrix: Dict[str, List[Any]] = {k: _as_list(v) for k, v in (raw.get("matrix") or {}).items()}
    if "np" not in matrix:
        matrix = {"np": _as_list(exp.mpi.get("np", 1)), **matrix}
    variants: List[Dict[str, Any]] = list(raw.get("variants") or [{}])
    excludes: List[Dict[str, Any]] = list(raw.get("exclude") or [])
    env_base = {str(k): v for k, v in (raw.get("env") or {}).items()}
    n_rep = int(repeats if repeats is not None else raw.get("repeats", 1))
    vary_seed = bool(raw.get("vary_seed", True))
    seed_base = int(raw.get("seed_base", 42))
    keys = list(matrix.keys())

    runs: List[Run] = []
    seen = set()
    for combo in itertools.product(*(matrix[k] for k in keys)):
        for var in variants:
            merged: Dict[str, Any] = {**base, **dict(zip(keys, combo)), **var}
            if any(all(format_value(merged.get(k)) == format_value(v) for k, v in ex.items()) for ex in excludes):
                continue
            for rep in range(n_rep):
                env_raw = {**env_base, **{k[4:]: v for k, v in merged.items() if k.startswith("env.")}}
                params = {k: v for k, v in merged.items() if not k.startswith("env.")}
                binary = str(params.pop("binary", exp.binary))
                if binary in SEEDED_BINARIES and "seed" not in params:
                    params["seed"] = seed_base + rep if vary_seed else seed_base
                np_raw = params.pop("np", 1)
                extra = {"repeat": rep, "binary": binary}
                np_val = int(_resolve({"np": np_raw}, {**extra})["np"]) if isinstance(np_raw, str) else int(np_raw)
                extra["np"] = np_val
                params = _resolve(params, extra)
                ns_env = {k.replace("-", "_"): v for k, v in params.items()}
                env = {k: format_value(v) or "" for k, v in _resolve(env_raw, {**extra, **ns_env}).items()}
                if filters:
                    view = {**{k: format_value(v) for k, v in params.items()}, "np": str(np_val),
                            "binary": binary, **{f"env.{k}": v for k, v in env.items()}}
                    if not all(view.get(fk) == fv for fk, fv in filters):
                        continue
                key = (binary, np_val, rep, tuple(sorted((k, format_value(v)) for k, v in params.items())),
                       tuple(sorted(env.items())))
                if key in seen:
                    continue
                seen.add(key)
                runs.append(Run(index=len(runs), name="", binary=binary, np=np_val, repeat=rep,
                                params=params, env=env))
    _name_runs(runs)
    return runs


def _view(run: Run) -> Dict[str, Any]:
    view: Dict[str, Any] = {"binary": run.binary, "np": run.np}
    view.update(run.params)
    view.update({f"env.{k}": v for k, v in run.env.items()})
    return view


def _name_runs(runs: List[Run]) -> None:
    """Sets `varying` (parameters that differ between runs, seed excluded) and run names."""
    views = [_view(r) for r in runs]
    order: List[str] = ["binary", "strategy", "np"]
    for v in views:
        order += [k for k in v if k not in order]
    varying = [k for k in order if k != "seed"
               and len({format_value(v.get(k)) for v in views}) > 1]
    for r, v in zip(runs, views):
        r.varying = {k: v.get(k) for k in varying}
        r.name = f"{r.index:04d}_{_slug(r.varying)}_r{r.repeat}"
