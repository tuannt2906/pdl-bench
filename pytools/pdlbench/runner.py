"""Experiment execution: builds mpirun command lines, runs them, records a manifest.

Layout of a run directory::

    results/<experiment>/<YYYYmmdd-HHMMSS>/
        config.yaml        copy of the experiment configuration
        manifest.json      environment + one entry per run (command, status, duration)
        raw/<run>.json     result file written by the binary (--output)
        logs/<run>.log     stdout + stderr of the run
        analysis/          written by `pdlbench aggregate` / `perfmodel`
        figures/           written by `pdlbench plot`
    results/<experiment>/latest -> <YYYYmmdd-HHMMSS>
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfgmod
from . import envinfo
from .util import dump_json, load_json, log

_MPICH_SINGLE_DASH = {"--bind-to", "--map-by", "--rank-by", "--hostfile", "--host"}


def _launcher_args(exp: cfgmod.Experiment, np_: int, flavor: str, phys: int, logical: int) -> List[str]:
    mpi = exp.mpi
    launcher = str(mpi.get("launcher", "mpirun"))
    extra = [str(a) for a in (mpi.get("extra_args") or [])]
    if flavor == "mpich":
        extra = [a[1:] if a in _MPICH_SINGLE_DASH else a for a in extra]
    if flavor == "openmpi":
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            extra.append("--allow-run-as-root")
        if np_ > phys:
            requested = extra[extra.index("--bind-to") + 1] if "--bind-to" in extra and \
                extra.index("--bind-to") + 1 < len(extra) else None
            if mpi.get("allow_oversubscribe", True) and requested == "none":
                # Explicitly unpinned (the OS places the ranks): keep it, just allow np > slots.
                extra.append("--oversubscribe")
                if "--map-by" in extra:
                    i = extra.index("--map-by")
                    del extra[i:i + 2]
            elif mpi.get("allow_oversubscribe", True):
                bind = "hwthread" if np_ <= logical else "none"
                if "--bind-to" in extra:
                    extra[extra.index("--bind-to") + 1] = bind
                if np_ <= logical:
                    extra.append("--use-hwthread-cpus")
                    if "--map-by" in extra and extra[extra.index("--map-by") + 1] == "core":
                        extra[extra.index("--map-by") + 1] = "hwthread"
                else:
                    extra.append("--oversubscribe")
                    if "--map-by" in extra:
                        i = extra.index("--map-by")
                        del extra[i:i + 2]
            else:
                log(f"warning: np={np_} exceeds {phys} physical cores and oversubscription is disabled")
    return [launcher, "-np", str(np_)] + extra


def build_command(exp: cfgmod.Experiment, run: cfgmod.Run, build_dir: Path, output: Optional[Path],
                  flavor: str, phys: int, logical: int) -> List[str]:
    binary = str(build_dir / "bin" / run.binary)
    args: List[str] = []
    for k, v in run.params.items():
        if k in cfgmod.RESERVED:
            continue
        sv = cfgmod.format_value(v)
        if sv is None:
            continue
        args += [f"--{k}", sv]
    if output is not None:
        args += ["--output", str(output)]
    if run.binary in cfgmod.SEEDED_BINARIES:
        args += ["--run-id", run.name, "--tag", exp.name]
    launch_single = bool(exp.mpi.get("launch_single", False))
    if run.np > 1 or launch_single:
        return _launcher_args(exp, run.np, flavor, phys, logical) + [binary] + args
    return [binary] + args


def _make_run_dir(results_dir: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    latest = results_dir / name / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(stamp, target_is_directory=True)
    except OSError:
        (results_dir / name / "LATEST").write_text(stamp + "\n", encoding="utf-8")
    return run_dir


def _parse_filters(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--filter expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _preflight_cuda(runs: List[cfgmod.Run], build_dir: Path) -> None:
    """Stops before any measurement if the experiment needs the GPU but CUDA does not work.

    Without this, a missing/shadowed driver library (e.g. a stale libcuda in /lib shadowing
    /usr/lib/wsl/lib) makes the binaries skip every GPU case silently and the experiment
    "succeeds" with CPU-only data.
    """
    def wants_gpu(r: cfgmod.Run) -> bool:
        vals = [str(v).lower() for v in r.params.values()]
        return any("cuda" in v or v in ("all", "gpu") for v in vals) and r.binary != "pdl_bench_comm"

    probe = build_dir / "bin" / "pdl_check"
    if not any(wants_gpu(r) for r in runs) or not probe.exists():
        return
    res = subprocess.run([str(probe), "--device", "cuda", "--tests", "kernels"],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0 or "CUDA requested but not available" in (res.stdout + res.stderr):
        raise SystemExit(
            "preflight: this experiment needs CUDA but the GPU check failed.\n"
            "  - run `build/bin/pdl_check --device cuda` to see the error\n"
            "  - WSL2: export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH\n"
            + (res.stdout + res.stderr)[-600:])
    log("preflight: CUDA kernels verified against the CPU reference")


def run_experiment(config_path: Path, build_dir: Path = Path("build"), results_dir: Path = Path("results"),
                   dry_run: bool = False, resume: Optional[Path] = None, repeats: Optional[int] = None,
                   filters: Optional[List[str]] = None, stop_on_error: bool = False) -> Optional[Path]:
    """Runs every expanded run of an experiment. Returns the run directory."""
    exp = cfgmod.load(config_path)
    runs = cfgmod.expand(exp, repeats=repeats, filters=_parse_filters(filters))
    launcher = str(exp.mpi.get("launcher", "mpirun"))
    flavor = envinfo.mpi_flavor(launcher) if shutil.which(launcher) else "openmpi"
    phys, logical = envinfo.physical_cores(), envinfo.logical_cpus()
    log(f"experiment '{exp.name}': {len(runs)} runs (mpi={flavor}, cores={phys}/{logical})")

    if dry_run:
        for r in runs:
            out = Path("<run_dir>/raw") / f"{r.name}.json"
            cmd = build_command(exp, r, build_dir, out, flavor, phys, logical)
            env = " ".join(f"{k}={v}" for k, v in r.env.items())
            print(f"[{r.index:04d}] {env + ' ' if env else ''}{shlex.join(cmd)}")
        return None

    missing = sorted({r.binary for r in runs if not (build_dir / "bin" / r.binary).exists()})
    if missing:
        raise SystemExit(f"missing binaries in {build_dir}/bin: {missing} (run scripts/build.sh)")

    _preflight_cuda(runs, build_dir)

    if resume is not None:
        run_dir = resume
        manifest = load_json(run_dir / "manifest.json") or {}
    else:
        run_dir = _make_run_dir(results_dir, exp.name)
        manifest = {}
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    if exp.path is not None and not (run_dir / "config.yaml").exists():
        shutil.copyfile(exp.path, run_dir / "config.yaml")

    manifest.setdefault("experiment", exp.name)
    manifest["description"] = exp.description
    manifest["config"] = exp.raw
    manifest["config_path"] = str(exp.path) if exp.path else None
    manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    manifest["environment"] = envinfo.collect(Path("."), launcher)
    entries: Dict[str, Dict[str, Any]] = {e["name"]: e for e in manifest.get("runs", [])}
    manifest_path = run_dir / "manifest.json"

    n_ok = n_fail = n_skip = 0
    t_start = time.time()
    for r in runs:
        out = run_dir / "raw" / f"{r.name}.json"
        cmd = build_command(exp, r, build_dir, out, flavor, phys, logical)
        entry = {
            "name": r.name, "index": r.index, "binary": r.binary, "np": r.np, "repeat": r.repeat,
            "params": r.params, "env": r.env, "varying": r.varying,
            "command": shlex.join(cmd), "output": str(out.relative_to(run_dir)),
        }
        if out.exists() and load_json(out) is not None:
            entry.update(entries.get(r.name, {}), status="ok")
            entries[r.name] = entry
            n_skip += 1
            continue
        log(f"[{r.index + 1}/{len(runs)}] {r.name}")
        env = dict(os.environ)
        env.update(r.env)
        t0 = time.time()
        status, rc = "ok", None
        with open(run_dir / "logs" / f"{r.name}.log", "w", encoding="utf-8") as logf:
            logf.write(f"# {entry['command']}\n# env: {r.env}\n")
            logf.flush()
            try:
                proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                      timeout=exp.timeout, check=False)
                rc = proc.returncode
                if rc != 0:
                    status = "failed"
                elif not out.exists():
                    status = "missing_output"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except OSError as exc:
                status = "failed"
                logf.write(f"\n# launch error: {exc}\n")
        entry.update(status=status, returncode=rc, duration_s=round(time.time() - t0, 3),
                     started=datetime.fromtimestamp(t0).isoformat(timespec="seconds"))
        entries[r.name] = entry
        manifest["runs"] = sorted(entries.values(), key=lambda e: e["index"])
        dump_json(manifest, manifest_path)
        if status == "ok":
            n_ok += 1
        else:
            n_fail += 1
            log(f"  -> {status} (rc={rc}); see {run_dir / 'logs' / (r.name + '.log')}")
            if stop_on_error:
                break
    manifest["runs"] = sorted(entries.values(), key=lambda e: e["index"])
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["summary"] = {"ok": n_ok, "failed": n_fail, "resumed": n_skip}
    dump_json(manifest, manifest_path)
    log(f"done in {time.time() - t_start:.1f}s: {n_ok} ok, {n_fail} failed, {n_skip} resumed -> {run_dir}")
    return run_dir


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="experiment YAML file (configs/*.yaml)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run directory")
    p.add_argument("--filter", action="append", metavar="KEY=VAL",
                   help="only runs whose parameter KEY equals VAL (repeatable; e.g. np=4, env.OMP_NUM_THREADS=1)")
    p.add_argument("--repeats", type=int, help="override the number of repetitions")
    p.add_argument("--build-dir", default="build", help="CMake build directory (binaries in <dir>/bin)")
    p.add_argument("--results-dir", default="results", help="root of result directories")
    p.add_argument("--stop-on-error", action="store_true", help="abort the experiment at the first failure")


def main(args: argparse.Namespace) -> int:
    run_dir = run_experiment(Path(args.config), Path(args.build_dir), Path(args.results_dir),
                             dry_run=args.dry_run, resume=Path(args.resume) if args.resume else None,
                             repeats=args.repeats, filters=args.filter, stop_on_error=args.stop_on_error)
    if run_dir is None:
        return 0
    print(run_dir)
    failed = int(((load_json(run_dir / "manifest.json") or {}).get("summary") or {}).get("failed", 0))
    return 1 if failed else 0
