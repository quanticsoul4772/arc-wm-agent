"""Build (and optionally push) the Kaggle code bundle dataset and kernel.

    python -m wmagent.kaggle.build --out RUN_DIR --label enf10 [--dry-run] ...

Writes the layout the replay scripts (probe/vast, probe/runpod) read:

    RUN_DIR/kaggle/source-dataset/   dataset-metadata.json, wmagent-bundle.json (marker),
                                     setup_vllm.py, run_config.json, agent_env.json, src/wmagent/...
    RUN_DIR/kaggle/kernel/           kernel-metadata.json, run.ipynb

Without --dry-run it then creates or versions the dataset, waits until Kaggle
reports it ready, and pushes the kernel, retrying while Kaggle rejects the push
because the GPU session cap is reached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import notebook
from .paths import BUNDLE_MARKER, COMPETITION_SLUG, MODEL_REF, WHEELHOUSE_REF

DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
MACHINE_SHAPE = "NvidiaRtxPro6000"
PROBE_GAMES = ["tn36-ef4dde99", "bp35-0a0ad940", "vc33-5430563c", "lp85-305b61c3", "tu93-0768757b", "sb26-7fbdac44"]
PACKAGE_DIR = Path(__file__).resolve().parents[1]  # wmagent/
SESSION_CAP_RE = re.compile(r"(?i)(maximum|limit|cap|quota)[^\n]{0,60}session|session[^\n]{0,60}(maximum|limit|cap|quota)")

AGENT_ENV_DEFAULTS: dict[str, str] = {
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_SEED": "-1",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}


def agent_env(a: argparse.Namespace) -> dict[str, str]:
    """Analyzer and gate env (spec 2.4); setup_vllm exports it next to the server URL and model id."""
    env = dict(AGENT_ENV_DEFAULTS)
    env.update({
        "PROBE_ENFORCE": str(a.probe_enforce), "PROBE_ACTIONS": str(a.probe_actions),
        "PROBE_MAX_REFUSALS": str(a.probe_max_refusals), "PROBE_RELEASE_AFTER_ACCEPT": str(a.probe_release_after_accept),
        "PROBE_OPTIONAL": str(a.probe_optional), "WM_MECHANISMS": a.wm_mechanisms or "base",
    })
    return env


def run_config(a: argparse.Namespace) -> dict[str, Any]:
    """The runner's config (spec 2.2); it loads with wmagent.runner.RunConfig.from_dict."""
    return {
        "label": a.label,
        "game_ids": a.games,
        "n_passes": a.n_passes,
        "concurrency": a.concurrency,
        "max_runtime_s_per_game": a.per_game_cap_s,
        "max_actions_per_game": a.max_actions_per_game,
        "analyzer_timeout_s": 900,
        "cancel_drain_timeout_s": 120,
        "max_runtime_s": 32400,
        "soft_deadline_buffer_s": 600,
        "environments_dir": f"/kaggle/input/competitions/{COMPETITION_SLUG}/environment_files",
    }


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_bundle(dst: Path, *, owner: str, slug: str, label: str, config: dict[str, Any], env: dict[str, str],
                 license_name: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PACKAGE_DIR, dst / "src" / "wmagent", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PACKAGE_DIR / "kaggle" / "setup_vllm.py", dst / "setup_vllm.py")
    (dst / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (dst / "agent_env.json").write_text(json.dumps(env, indent=2) + "\n")
    (dst / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{owner}/{slug}", "licenses": [{"name": license_name}]}, indent=2) + "\n")
    files = {p.relative_to(dst).as_posix(): _sha256(p) for p in sorted(dst.rglob("*")) if p.is_file()}
    (dst / BUNDLE_MARKER).write_text(json.dumps({
        "slug": slug, "label": label, "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }, indent=2) + "\n")


def kernel_metadata(*, owner: str, kernel_slug: str, bundle_ref: str, private: bool) -> dict[str, Any]:
    return {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [bundle_ref, WHEELHOUSE_REF, MODEL_REF],
        "competition_sources": [COMPETITION_SLUG],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": MACHINE_SHAPE,
        "docker_image": DOCKER_IMAGE,
    }


def write_kernel(dst: Path, *, meta: dict[str, Any], bundle_slug: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dst / "run.ipynb").write_text(json.dumps(notebook.render(bundle_slug), indent=1) + "\n")


def build(a: argparse.Namespace) -> tuple[Path, Path]:
    base = Path(a.out) / "kaggle"
    src_dir, kernel_dir = base / "source-dataset", base / "kernel"
    write_bundle(src_dir, owner=a.owner, slug=a.slug, label=a.label, config=run_config(a), env=agent_env(a),
                 license_name=a.license)
    write_kernel(kernel_dir, meta=kernel_metadata(owner=a.owner, kernel_slug=a.kernel_slug,
                                                  bundle_ref=f"{a.owner}/{a.slug}", private=not a.public),
                 bundle_slug=a.slug)
    return src_dir, kernel_dir


Runner = Callable[..., subprocess.CompletedProcess]


def _kaggle(run: Runner, *args: str) -> subprocess.CompletedProcess:
    return run(["kaggle", *args], capture_output=True, text=True)


def push(src_dir: Path, kernel_dir: Path, ref: str, *, run: Runner = subprocess.run,
         sleep: Callable[[float], None] = time.sleep, ready_timeout_s: float = 900, push_retries: int = 24,
         retry_wait_s: float = 300) -> None:
    status = _kaggle(run, "datasets", "status", ref)
    exists = status.returncode == 0 and "404" not in (status.stdout + status.stderr)
    if exists:
        r = _kaggle(run, "datasets", "version", "-p", str(src_dir), "-m", f"build {datetime.now(UTC):%F %T}",
                    "--dir-mode", "zip")
    else:
        r = _kaggle(run, "datasets", "create", "-p", str(src_dir), "--dir-mode", "zip")
    if r.returncode != 0:
        raise RuntimeError(f"dataset upload failed: {r.stdout}{r.stderr}")
    waited = 0.0
    while True:
        s = _kaggle(run, "datasets", "status", ref)
        if "ready" in s.stdout.lower():
            break
        if waited >= ready_timeout_s:
            raise TimeoutError(f"dataset {ref} not ready after {ready_timeout_s:.0f}s: {s.stdout}{s.stderr}")
        sleep(15)
        waited += 15
    for attempt in range(push_retries + 1):
        r = _kaggle(run, "kernels", "push", "-p", str(kernel_dir))
        out = r.stdout + r.stderr
        if SESSION_CAP_RE.search(out):
            if attempt == push_retries:
                raise RuntimeError(f"kernel push still rejected by the session cap after {push_retries} retries: {out}")
            print(f"[build] session cap reached; retry {attempt + 1}/{push_retries} in {retry_wait_s:.0f}s", flush=True)
            sleep(retry_wait_s)
            continue
        if r.returncode != 0 or "error" in out.lower():
            raise RuntimeError(f"kernel push failed: {out}")
        print(f"[build] {out.strip()}", flush=True)
        return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run dir; files go under <out>/kaggle/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--owner", default="quanticsoul4772")
    ap.add_argument("--slug", default=None, help="code bundle dataset slug (default wm-src-<label>)")
    ap.add_argument("--kernel-slug", default=None, help="kernel slug (default wm-run-<label>)")
    ap.add_argument("--games", default=",".join(PROBE_GAMES), type=lambda s: [g for g in s.split(",") if g])
    ap.add_argument("--n-passes", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--per-game-cap-s", type=float, default=2700)
    ap.add_argument("--max-actions-per-game", type=int, default=None)
    ap.add_argument("--probe-enforce", type=int, default=1)
    ap.add_argument("--probe-actions", type=int, default=8)
    ap.add_argument("--probe-max-refusals", type=int, default=8)
    ap.add_argument("--probe-release-after-accept", type=int, default=0)
    ap.add_argument("--probe-optional", type=int, default=0)
    ap.add_argument("--wm-mechanisms", choices=["base", "m1a", "all", "all-nocont", "all-nocont-fb"], default=None,
                    help="Plan C arm: base (as before), m1a (capture-on-deny + stated rule), all, all-nocont (all without "
                    "the yield-continuation prompt), all-nocont-fb (all-nocont plus the verdict after every cell "
                    "that defines predict). Required for "
                    "a push: any pushed kernel can become a submission, and a missing arm would silently ship "
                    "base. A --dry-run without it writes base.")
    ap.add_argument("--license", default="unknown", help="Kaggle dataset license name")
    ap.add_argument("--public", action="store_true", help="make the kernel public (default private)")
    ap.add_argument("--dry-run", action="store_true", help="only write files")
    a = ap.parse_args(argv)
    a.slug = a.slug or f"wm-src-{a.label}"
    a.kernel_slug = a.kernel_slug or f"wm-run-{a.label}"
    return a


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if a.wm_mechanisms is None and not a.dry_run:
        raise SystemExit("[build] --wm-mechanisms is required for a push (base, m1a, all, all-nocont or all-nocont-fb): any pushed kernel can "
                         "become a submission, and a missing arm would silently ship base")
    src_dir, kernel_dir = build(a)
    print(f"[build] bundle {src_dir}\n[build] kernel {kernel_dir}\n[build] WM_MECHANISMS={a.wm_mechanisms or 'base'}",
          flush=True)
    if a.dry_run:
        return
    push(src_dir, kernel_dir, f"{a.owner}/{a.slug}")


if __name__ == "__main__":
    main(sys.argv[1:])
