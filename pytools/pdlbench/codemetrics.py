"""Source-code metrics for the AI coding-agent evaluation (docs/AI_AGENT_EVALUATION.md).

For each source tree: files and lines (code / comment / blank) per language, number of
functions with mean / max length, TODO/FIXME markers, MPI and CUDA API usage counts,
compiler warnings from a build log and, when `lizard` is installed, cyclomatic
complexity. Two trees can be compared side by side (AI-generated vs. team-written).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LANGS = {".c": "C", ".h": "C", ".cu": "CUDA", ".cuh": "CUDA", ".py": "Python", ".cmake": "CMake",
         ".sh": "Shell", ".yaml": "YAML", ".yml": "YAML"}
SKIP_DIRS = {".git", "build", "results", "data", "__pycache__", ".venv", "venv", "node_modules"}
C_FUNC = re.compile(r"^[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(\{)?\s*$")
C_KEYWORDS = {"if", "for", "while", "switch", "return", "else", "do", "sizeof"}
MPI_CALL = re.compile(r"\bMPI_[A-Z][a-z_]+\s*\(")
CUDA_CALL = re.compile(r"\b(cuda[A-Z]\w*|cublas[A-Z]\w*)\s*\(|<<<")


def _lang(path: Path) -> Optional[str]:
    if path.name == "CMakeLists.txt":
        return "CMake"
    return LANGS.get(path.suffix.lower())


def _count_lines(text: str, lang: str) -> Dict[str, int]:
    code = comment = blank = 0
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            blank += 1
            continue
        if lang in ("C", "CUDA"):
            if in_block:
                comment += 1
                if "*/" in line:
                    in_block = False
                continue
            if line.startswith("//"):
                comment += 1
                continue
            if line.startswith("/*"):
                comment += 1
                in_block = "*/" not in line
                continue
        elif lang in ("Python", "Shell", "CMake", "YAML") and line.startswith("#"):
            comment += 1
            continue
        code += 1
    return {"code": code, "comment": comment, "blank": blank}


def _c_functions(text: str) -> List[int]:
    """Lengths (lines) of top-level C/CUDA function definitions."""
    lines = text.splitlines()
    lengths: List[int] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = C_FUNC.match(line) if line and not line[0].isspace() else None
        if m and m.group(1) not in C_KEYWORDS and not line.startswith(("#", "typedef", "static_assert")):
            j = i if m.group(2) else i + 1
            if j < len(lines) and "{" in lines[j]:
                depth, k = 0, j
                while k < len(lines):
                    depth += lines[k].count("{") - lines[k].count("}")
                    if depth <= 0 and k > j or (depth == 0 and "}" in lines[k]):
                        break
                    k += 1
                lengths.append(k - i + 1)
                i = k + 1
                continue
        i += 1
    return lengths


def _py_functions(text: str) -> List[int]:
    lines = text.splitlines()
    starts = [(i, len(l) - len(l.lstrip())) for i, l in enumerate(lines) if l.lstrip().startswith("def ")]
    lengths = []
    for i, indent in starts:
        end = i + 1
        while end < len(lines):
            l = lines[end]
            if l.strip() and len(l) - len(l.lstrip()) <= indent:
                break
            end += 1
        lengths.append(end - i)
    return lengths


def _lizard(root: Path) -> Optional[Dict[str, float]]:
    if shutil.which("lizard") is None:
        return None
    try:
        out = subprocess.run(["lizard", "--csv", str(root)], capture_output=True, text=True, timeout=300,
                             check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    ccn = [int(r.split(",")[1]) for r in out.splitlines() if r.count(",") > 5 and r.split(",")[1].isdigit()]
    if not ccn:
        return None
    return {"functions": len(ccn), "ccn_mean": sum(ccn) / len(ccn), "ccn_max": max(ccn),
            "ccn_over_10": sum(1 for c in ccn if c > 10)}


def count_warnings(build_log: Optional[Path]) -> Optional[Dict[str, int]]:
    if build_log is None or not build_log.exists():
        return None
    text = build_log.read_text(encoding="utf-8", errors="replace")
    return {"warnings": len(re.findall(r"\bwarning\b[:\s]", text, re.I)),
            "errors": len(re.findall(r"\berror\b[:\s]", text, re.I))}


def analyze_tree(root: Path, build_log: Optional[Path] = None) -> Dict[str, Any]:
    per_lang: Dict[str, Dict[str, int]] = {}
    func_lengths: List[int] = []
    todo = mpi_calls = cuda_calls = 0
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        lang = _lang(path)
        if lang is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        files += 1
        stats = per_lang.setdefault(lang, {"files": 0, "code": 0, "comment": 0, "blank": 0})
        stats["files"] += 1
        for k, v in _count_lines(text, lang).items():
            stats[k] += v
        todo += len(re.findall(r"\b(TODO|FIXME|XXX|HACK)\b", text))
        if lang in ("C", "CUDA"):
            func_lengths += _c_functions(text)
            mpi_calls += len(MPI_CALL.findall(text))
            cuda_calls += len(CUDA_CALL.findall(text))
        elif lang == "Python":
            func_lengths += _py_functions(text)
    code = sum(s["code"] for s in per_lang.values())
    comment = sum(s["comment"] for s in per_lang.values())
    return {
        "root": str(root), "files": files, "languages": per_lang,
        "code_lines": code, "comment_lines": comment,
        "comment_ratio": comment / (code + comment) if code + comment else 0.0,
        "functions": len(func_lengths),
        "function_length_mean": sum(func_lengths) / len(func_lengths) if func_lengths else 0.0,
        "function_length_max": max(func_lengths) if func_lengths else 0,
        "functions_over_60_lines": sum(1 for n in func_lengths if n > 60),
        "todo_markers": todo, "mpi_call_sites": mpi_calls, "cuda_call_sites": cuda_calls,
        "build": count_warnings(build_log), "complexity": _lizard(root),
    }


def to_markdown(results: Sequence[Dict[str, Any]], names: Sequence[str]) -> str:
    keys = ["files", "code_lines", "comment_lines", "comment_ratio", "functions", "function_length_mean",
            "function_length_max", "functions_over_60_lines", "todo_markers", "mpi_call_sites", "cuda_call_sites"]
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in keys:
        vals = [r.get(k) for r in results]
        lines.append(f"| {k} | " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals) + " |")
    for sub, fields in (("build", ("warnings", "errors")), ("complexity", ("ccn_mean", "ccn_max", "ccn_over_10"))):
        for f in fields:
            vals = [(r.get(sub) or {}).get(f) for r in results]
            if any(v is not None for v in vals):
                lines.append(f"| {sub}.{f} | " + " | ".join("–" if v is None else
                                                          (f"{v:.3g}" if isinstance(v, float) else str(v))
                                                          for v in vals) + " |")
    langs = sorted({lang for r in results for lang in r["languages"]})
    for lang in langs:
        vals = [(r["languages"].get(lang) or {}).get("code", 0) for r in results]
        lines.append(f"| LOC {lang} | " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines) + "\n"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("src_dirs", nargs="+", help="one or two source trees (e.g. AI-generated vs team)")
    p.add_argument("--build-log", nargs="+", help="build logs (same order as src_dirs) for warning counts")
    p.add_argument("--names", nargs="+", help="column names for the comparison table")
    p.add_argument("--out", default="results/ai_eval", help="output directory for metrics.json / metrics.md")


def main(args: argparse.Namespace) -> int:
    roots = [Path(d) for d in args.src_dirs]
    logs = [Path(p) for p in (args.build_log or [])] + [None] * len(roots)
    names = list(args.names or [r.name or str(r) for r in roots])[:len(roots)]
    results = [analyze_tree(r, logs[i]) for i, r in enumerate(roots)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(dict(zip(names, results)), indent=2) + "\n", encoding="utf-8")
    md = to_markdown(results, names)
    (out / "metrics.md").write_text(md, encoding="utf-8")
    print(md)
    return 0
