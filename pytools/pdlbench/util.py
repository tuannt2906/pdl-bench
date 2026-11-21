"""Shared helpers: JSON IO, path handling, statistics and logging."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Two-sided 95% Student-t critical values for df = 1..30.
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


def t95(df: int) -> float:
    """Two-sided 95% t critical value (normal approximation above df=30)."""
    if df <= 0:
        return float("nan")
    if df <= len(_T95):
        return _T95[df - 1]
    return 1.96 + 2.4 / df  # smooth approach to the normal quantile


def ci95(std: float, n: int) -> float:
    """Half-width of the 95% confidence interval of a mean."""
    if n is None or n < 2 or std is None or not math.isfinite(std):
        return float("nan")
    return t95(n - 1) * std / math.sqrt(n)


def log(msg: str) -> None:
    print(f"[pdlbench] {msg}", file=sys.stderr, flush=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON file; returns None (and warns) if missing or malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"warning: cannot read {path}: {exc}")
        return None


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)



def get(d: Optional[Dict[str, Any]], *path: str, default: Any = None) -> Any:
    """Safe nested lookup: get(d, "summary", "phases_mean", "comm")."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return default if cur is None else cur


def find_raw_jsons(run_dir: Path) -> List[Path]:
    """Result JSONs of a run directory (`raw/*.json`), or of a plain directory."""
    raw = run_dir / "raw"
    base = raw if raw.is_dir() else run_dir
    return sorted(p for p in base.glob("*.json") if p.name != "manifest.json")


def resolve_run_dirs(paths: Sequence[str]) -> List[Path]:
    """Expands experiment dirs (results/<exp>) to their `latest` run dir."""
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if (path / "latest").exists() and not (path / "raw").is_dir():
            path = (path / "latest").resolve()
        if not path.exists():
            log(f"warning: {path} does not exist, skipped")
            continue
        out.append(path)
    return out



def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"



def fmt_value(v: Any) -> str:
    """Compact label formatting: integral floats print as integers."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
