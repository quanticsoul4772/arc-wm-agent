"""Locate Kaggle inputs by search.

Kaggle has mounted datasets at /kaggle/input/<slug> and at
/kaggle/input/datasets/<owner>/<slug>, and the competition at
/kaggle/input/competitions/<slug> or /kaggle/input/<slug>. Each finder checks
the known locations first, then walks the input root (bounded depth) for a
directory that satisfies a content predicate, so a new mount layout still
resolves.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

COMPETITION_SLUG = "arc-prize-2026-arc-agi-3"
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL_REF = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
BUNDLE_MARKER = "wmagent-bundle.json"

# Directories never worth descending into while searching.
_PRUNE = {"environment_files", "arc_agi_3_wheels", "__pycache__", ".git", "ARC-AGI-3-Agents", "src"}


def input_root() -> Path:
    return Path(os.environ.get("WMAGENT_KAGGLE_INPUT", "/kaggle/input"))


def working_root() -> Path:
    return Path(os.environ.get("WMAGENT_KAGGLE_WORKING", "/kaggle/working"))


def candidate_paths(root: Path, ref: str) -> list[Path]:
    """Known mount points for a dataset ref `owner/slug` (or a bare competition slug)."""
    owner, _, slug = ref.rpartition("/")
    out = [root / slug, root / "competitions" / slug]
    if owner:
        out.insert(1, root / "datasets" / owner / slug)
    return out


def walk_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    """Breadth-first directories under root (root included), skipping pruned names."""
    frontier = [root]
    for _ in range(max_depth + 1):
        nxt: list[Path] = []
        for d in frontier:
            yield d
            try:
                children = sorted(p for p in d.iterdir() if p.is_dir())
            except OSError:
                continue
            nxt.extend(c for c in children if c.name not in _PRUNE)
        frontier = nxt


def find_dir(root: Path, ref: str, ok: Callable[[Path], bool], what: str, max_depth: int = 5) -> Path:
    """Known locations for `ref` first, then any directory under root satisfying `ok`.

    Among search hits, one whose path contains the slug wins over others.
    """
    for p in candidate_paths(root, ref):
        if p.is_dir() and ok(p):
            return p
    slug = ref.rpartition("/")[2]
    hits = [d for d in walk_dirs(root, max_depth) if ok(d)]
    if not hits:
        raise FileNotFoundError(f"{what} ({ref}) not found under {root}")
    hits.sort(key=lambda d: (slug not in str(d), len(d.parts), str(d)))
    return hits[0]


def is_competition_dir(d: Path) -> bool:
    return (d / "environment_files").is_dir() and (d / "arc_agi_3_wheels").is_dir()


def is_wheelhouse_dir(d: Path) -> bool:
    lock = d / "requirements.lock"
    if not lock.is_file():
        return False
    try:
        return any(line.strip().startswith("vllm==") for line in lock.read_text().splitlines())
    except OSError:
        return False


def is_model_dir(d: Path) -> bool:
    return (d / "config.json").is_file() and any(d.glob("*.safetensors"))


def is_bundle_dir(d: Path) -> bool:
    return (d / BUNDLE_MARKER).is_file()


def find_competition(root: Path | None = None) -> Path:
    return find_dir(root or input_root(), COMPETITION_SLUG, is_competition_dir, "competition data")


def find_wheelhouse(root: Path | None = None) -> Path:
    return find_dir(root or input_root(), WHEELHOUSE_REF, is_wheelhouse_dir, "vLLM wheelhouse")


def find_model(root: Path | None = None) -> Path:
    return find_dir(root or input_root(), MODEL_REF, is_model_dir, "model snapshot")


def find_bundle(slug: str = "", root: Path | None = None) -> Path:
    return find_dir(root or input_root(), slug, is_bundle_dir, "code bundle")
