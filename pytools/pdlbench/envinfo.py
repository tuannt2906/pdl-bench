"""Host environment capture (complements the `environment` object written by the C binaries).

Collected: OS / kernel, CPU topology (lscpu, physical cores), memory, GPU (nvidia-smi),
toolchain versions (gcc, nvcc, cmake, mpirun), git state and Python packages.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run(cmd: List[str], timeout: float = 15.0) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        text = (out.stdout or "") + (out.stderr or "")
        return text.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _first_line(text: Optional[str]) -> Optional[str]:
    return text.splitlines()[0].strip() if text else None


def physical_cores() -> int:
    """Number of physical cores (unique (package, core id) pairs on Linux)."""
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    out = _run(["sysctl", "-n", "hw.physicalcpu"])
    if out and out.strip().isdigit():
        return int(out.strip())
    return os.cpu_count() or 1


def logical_cpus() -> int:
    return os.cpu_count() or 1


def mpi_flavor(launcher: str = "mpirun") -> str:
    """Returns 'openmpi', 'mpich', 'intel' or 'unknown'."""
    text = _run([launcher, "--version"]) or ""
    low = text.lower()
    if "open mpi" in low or "open-mpi" in low or "openrte" in low:
        return "openmpi"
    if "intel" in low:
        return "intel"
    if "hydra" in low or "mpich" in low:
        return "mpich"
    return "unknown"


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or None


def _mem_total() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    out = _run(["sysctl", "-n", "hw.memsize"])
    return int(out) if out and out.isdigit() else None


def _gpus() -> List[Dict[str, str]]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,pcie.link.gen.max",
                "--format=csv,noheader"])
    gpus = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2],
                             "max_sm_clock": parts[3] if len(parts) > 3 else "",
                             "pcie_gen": parts[4] if len(parts) > 4 else ""})
    return gpus


def _git(repo: Path) -> Dict[str, Any]:
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo), "status", "--porcelain"])
    return {"commit": commit if commit and " " not in commit else None,
            "dirty": bool(status) if commit else None}


def _python_packages() -> Dict[str, str]:
    pkgs = {}
    for name in ("numpy", "pandas", "matplotlib", "yaml"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "?")
        except ImportError:
            pkgs[name] = "missing"
    return pkgs


def collect(repo: Path = Path("."), launcher: str = "mpirun") -> Dict[str, Any]:
    uname = platform.uname()
    is_wsl = "microsoft" in uname.release.lower()
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": uname.node,
        "os": f"{uname.system} {uname.release} {uname.machine}",
        "wsl": is_wsl,
        "cpu_model": _cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cpus": logical_cpus(),
        "memory_bytes": _mem_total(),
        "lscpu": _run(["lscpu"]),
        "gpus": _gpus(),
        "nvidia_smi": _first_line(_run(["nvidia-smi", "-L"])),
        "gcc": _first_line(_run(["gcc", "--version"])),
        "nvcc": (_run(["nvcc", "--version"]) or "").splitlines()[-1:] or None,
        "cmake": _first_line(_run(["cmake", "--version"])),
        "mpirun": _first_line(_run([launcher, "--version"])),
        "mpi_flavor": mpi_flavor(launcher),
        "git": _git(repo),
        "python": sys.version.split()[0],
        "python_packages": _python_packages(),
        "omp_env": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "OMPI_", "CUDA_"))},
    }


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="-", help="output JSON file ('-' = stdout)")
    p.add_argument("--launcher", default="mpirun", help="MPI launcher to query")


def main(args: argparse.Namespace) -> int:
    info = collect(Path("."), args.launcher)
    text = json.dumps(info, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0
