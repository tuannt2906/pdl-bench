"""Dataset preparation: download MNIST / CIFAR-10 and convert them to `.pdlds`.

The `.pdlds` format (see include/pdl/dataset.h) is:

    char[8]  magic "PDLDS1\\0\\0"
    uint32   n, c, h, w, num_classes, reserved=0     (little endian)
    uint8    pixels[n][c][h][w]
    uint8    labels[n]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .util import log

MAGIC = b"PDLDS1\x00\x00"
HEADER = struct.Struct("<8s6I")

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
MNIST_FILES: Dict[str, str] = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


# --------------------------------------------------------------------------- io

def write_pdlds(path: Path, pixels: np.ndarray, labels: np.ndarray, num_classes: int) -> None:
    """Writes uint8 pixels [n, c, h, w] and uint8 labels [n] to `path`."""
    if pixels.ndim != 4:
        raise ValueError("pixels must have shape [n, c, h, w]")
    n, c, h, w = pixels.shape
    if labels.shape != (n,):
        raise ValueError("labels must have shape [n]")
    if labels.max(initial=0) >= num_classes:
        raise ValueError("label out of range")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, n, c, h, w, num_classes, 0))
        f.write(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(labels, dtype=np.uint8).tobytes())
    tmp.replace(path)
    log(f"wrote {path} (n={n}, shape={c}x{h}x{w}, classes={num_classes})")


def read_header(path: Path) -> Dict[str, int]:
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, n, c, h, w, k, _ = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    expected = HEADER.size + n * c * h * w + n
    actual = path.stat().st_size
    return {"n": n, "c": c, "h": h, "w": w, "num_classes": k,
            "file_bytes": actual, "expected_bytes": expected, "valid": int(actual == expected)}


def read_pdlds(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    hdr = read_header(path)
    n, c, h, w = hdr["n"], hdr["c"], hdr["h"], hdr["w"]
    data = np.fromfile(path, dtype=np.uint8, offset=HEADER.size)
    pixels = data[: n * c * h * w].reshape(n, c, h, w)
    labels = data[n * c * h * w: n * c * h * w + n]
    return pixels, labels, hdr["num_classes"]


# --------------------------------------------------------------------- download

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(urls: List[str], dest: Path, md5: Optional[str], verify: bool) -> Path:
    if dest.exists() and (not verify or md5 is None or _md5(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            log(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "pdlbench-data/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            if verify and md5 is not None and _md5(dest) != md5:
                raise IOError(f"checksum mismatch for {dest.name}")
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = exc
            log(f"  failed: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"could not download {dest.name}: {last_err}")


def _parse_idx(raw: bytes) -> np.ndarray:
    zero, dtype, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError("unsupported IDX file")
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def prepare_mnist(out_dir: Path, raw_dir: Path, force: bool, verify: bool) -> None:
    targets = {"train": out_dir / "mnist-train.pdlds", "test": out_dir / "mnist-test.pdlds"}
    if not force and all(p.exists() for p in targets.values()):
        log("MNIST already prepared (use --force to rebuild)")
        return
    files = {}
    for name, md5 in MNIST_FILES.items():
        files[name] = _download([m + name for m in MNIST_MIRRORS], raw_dir / name, md5, verify)
    for split, prefix in (("train", "train"), ("test", "t10k")):
        images = _parse_idx(gzip.decompress(files[f"{prefix}-images-idx3-ubyte.gz"].read_bytes()))
        labels = _parse_idx(gzip.decompress(files[f"{prefix}-labels-idx1-ubyte.gz"].read_bytes()))
        expected = 60000 if split == "train" else 10000
        if images.shape != (expected, 28, 28) or labels.shape != (expected,):
            raise ValueError(f"unexpected MNIST {split} shape {images.shape}")
        write_pdlds(targets[split], images[:, None, :, :], labels, 10)


def prepare_cifar10(out_dir: Path, raw_dir: Path, force: bool, verify: bool) -> None:
    targets = {"train": out_dir / "cifar10-train.pdlds", "test": out_dir / "cifar10-test.pdlds"}
    if not force and all(p.exists() for p in targets.values()):
        log("CIFAR-10 already prepared (use --force to rebuild)")
        return
    archive = _download([CIFAR10_URL], raw_dir / "cifar-10-binary.tar.gz", CIFAR10_MD5, verify)
    batches: Dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name.endswith(".bin") and ("data_batch" in name or "test_batch" in name):
                fobj = tar.extractfile(member)
                if fobj is not None:
                    batches[name] = fobj.read()
    record = 1 + 3 * 32 * 32

    def decode(names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        buf = b"".join(batches[n] for n in names)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, record)
        return arr[:, 1:].reshape(-1, 3, 32, 32), arr[:, 0].copy()

    train_names = [f"data_batch_{i}.bin" for i in range(1, 6)]
    missing = [n for n in train_names + ["test_batch.bin"] if n not in batches]
    if missing:
        raise ValueError(f"CIFAR-10 archive is missing {missing}")
    x, y = decode(train_names)
    write_pdlds(targets["train"], x, y, 10)
    x, y = decode(["test_batch.bin"])
    write_pdlds(targets["test"], x, y, 10)


# -------------------------------------------------------------------------- cli

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", nargs="+", default=["mnist"], choices=["mnist", "cifar10", "all"],
                   help="datasets to prepare (default: mnist)")
    p.add_argument("--out", default="data", help="output directory for .pdlds files")
    p.add_argument("--raw", default=None, help="download cache (default: <out>/raw)")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    p.add_argument("--no-verify", action="store_true", help="skip MD5 verification of downloads")
    p.add_argument("--info", nargs="+", metavar="FILE", help="print the header of .pdlds files and exit")


def main(args: argparse.Namespace) -> int:
    if args.info:
        rc = 0
        for f in args.info:
            try:
                hdr = read_header(Path(f))
                status = "ok" if hdr["valid"] else "SIZE MISMATCH"
                print(f"{f}: n={hdr['n']} c={hdr['c']} h={hdr['h']} w={hdr['w']} "
                      f"classes={hdr['num_classes']} bytes={hdr['file_bytes']} [{status}]")
                rc |= 0 if hdr["valid"] else 1
            except (OSError, ValueError) as exc:
                print(f"{f}: error: {exc}")
                rc = 1
        return rc
    out = Path(args.out)
    raw = Path(args.raw) if args.raw else out / "raw"
    names = {"mnist", "cifar10"} if "all" in args.dataset else set(args.dataset)
    verify = not args.no_verify
    if "mnist" in names:
        prepare_mnist(out, raw, args.force, verify)
    if "cifar10" in names:
        prepare_cifar10(out, raw, args.force, verify)
    return 0

