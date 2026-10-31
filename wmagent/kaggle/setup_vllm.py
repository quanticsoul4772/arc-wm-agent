"""Install and start the local vLLM OpenAI server for a Kaggle run (stdlib only).

The builder copies this file to the code bundle root; the notebook loads it with
runpy, so it must not import the wmagent package. `start()` returns a handle;
`teardown(handle)` stops the server process group and removes the installed
site-packages.

Sequence (spec 2.6 step 4): GPU check, model copy to local disk, wheelhouse
install (reused when the stamp matches), server launch, /v1/models health poll, one smoke chat, and the
analyzer env exported into os.environ.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

STAMP = "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6"
STAMP_FILE = ".wmagent-wheelhouse-stamp"
SERVED_MODEL = "vrfai/Qwen3.6-27B-FP8"
HOST = "127.0.0.1"
PORT = 1234
HEALTH_POLL_S = 5.0
HEALTH_TIMEOUT_S = 2400.0
HEALTH_PROGRESS_S = 60.0  # wait_healthy prints a line this often (it was silent for up to 40 minutes)
INSTALL_TIMEOUT_S = 1800.0  # the wheelhouse install had no bound
WATCH_POLL_S = 30.0
# WM_SPEC_MTP_TOKENS > 0: speculative decoding with the model's multi-token-prediction head (paths.MTP_REF),
# that many draft tokens per step. 0 (default) is the server as before.
SPEC_ENV = "WM_SPEC_MTP_TOKENS"
SPEC_MIN_ACCEPTANCE = 1.3  # mean accepted tokens per step on the counting check; ~1.0 means the head did not load
SPEC_CHECK_TIMEOUT_S = 90.0
WATCH_GPU_S = 600.0
# vLLM's load of the 35.9 GB snapshot straight from the gcsfuse input mount did not finish in 2,400 s
# (wm-run-paper-check v1/v2), while one sequential read ran at 45 MB/s (wm-model-read-test, 2026-10-24).
STAGE_DIR = Path("/tmp/wmagent-model")
COPY_CHUNK = 64 << 20
COPY_REPORT = 4 << 30
KILL_GRACE_S = 30.0
EXPECTED_GPU = "rtx pro 6000"


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    log_path: Path
    pid_path: Path
    site_packages: Path | None
    base_url: str
    ready_s: float = 0.0
    torn_down: bool = field(default=False)


def check_gpu(expected: str = EXPECTED_GPU, count: int = 1) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout
    names = [n.strip() for n in out.splitlines() if n.strip()]
    matching = [n for n in names if expected in n.lower()]
    if len(names) != count or len(matching) != count:
        raise RuntimeError(f"expected {count} x '{expected}' GPU, nvidia-smi reports {names}")
    return names


def install_wheelhouse(wheelhouse: Path, target: Path) -> bool:
    """Install the wheelhouse into `target`. Returns False when the stamp says it is already there."""
    stamp = target / STAMP_FILE
    if stamp.is_file() and stamp.read_text().strip() == STAMP:
        return False
    cmd = [
        sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse),
        "-r", str(wheelhouse / "requirements.lock"), "--target", str(target),
        "--only-binary", ":all:", "--no-compile", "--ignore-installed", "--upgrade",
    ]
    subprocess.run(cmd, check=True, timeout=INSTALL_TIMEOUT_S)
    stamp.write_text(STAMP + "\n")
    return True


def stage_model(model_dir: Path, dest: Path = STAGE_DIR) -> Path:
    """Copy the model directory to local disk with sequential reads; returns `dest`."""
    files = sorted(f for f in model_dir.rglob("*") if f.is_file())
    need = sum(f.stat().st_size for f in files)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < need:
        raise RuntimeError(f"{dest} has {free / 1e9:.1f} GB free, the model needs {need / 1e9:.1f} GB")
    t0 = time.monotonic()
    done, mark = 0, COPY_REPORT
    for f in files:
        out = dest / f.relative_to(model_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "rb", buffering=0) as src, open(out, "wb") as dst:
            while chunk := src.read(COPY_CHUNK):
                dst.write(chunk)
                done += len(chunk)
                if done >= mark:
                    dt = time.monotonic() - t0
                    print(f"[setup_vllm] model copy {done / 1e9:.1f}/{need / 1e9:.1f} GB in {dt:.0f}s "
                          f"({done / dt / 1e6:.0f} MB/s)", flush=True)
                    mark += COPY_REPORT
    print(f"[setup_vllm] model copied to {dest}: {need / 1e9:.1f} GB in {time.monotonic() - t0:.0f}s", flush=True)
    return dest


def spec_tokens(agent_env: dict[str, str]) -> int:
    raw = str(agent_env.get(SPEC_ENV, "0")).strip() or "0"
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{SPEC_ENV}={raw!r} is not an integer") from None
    if value < 0:
        raise ValueError(f"{SPEC_ENV}={value} must be >= 0")
    return value


def add_mtp_head(staged: Path, mtp_dir: Path) -> None:
    """Copy the BF16 MTP head next to the FP8 weights and keep it out of the FP8 scheme (compressed-tensors
    `ignore`), so vLLM loads it unquantized."""
    shutil.copyfile(mtp_dir / "mtp.safetensors", staged / "mtp.safetensors")
    config_path = staged / "config.json"
    config = json.loads(config_path.read_text())
    ignore = config.setdefault("quantization_config", {}).setdefault("ignore", [])
    if "re:.*mtp.*" not in ignore:
        ignore.append("re:.*mtp.*")
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[setup_vllm] MTP head grafted from {mtp_dir}", flush=True)


def vllm_command(model_dir: Path, host: str = HOST, port: int = PORT, spec: int = 0) -> list[str]:
    extra = ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": spec})] if spec else []
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir), "--served-model-name", SERVED_MODEL,
        "--host", host, "--port", str(port), "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--generation-config", "vllm", "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
        "--max-model-len", "65536",
        *extra,
    ]


def server_env(site_packages: Path | None, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if site_packages is not None:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(site_packages), env.get("PYTHONPATH", "")) if p)
    env.update({
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "VLLM_NO_USAGE_STATS": "1",
    })
    return env


def _get(url: str, timeout: float = 5.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def _log_tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def wait_healthy(handle: ServerHandle, timeout_s: float = HEALTH_TIMEOUT_S, poll_s: float = HEALTH_POLL_S,
                 progress_s: float = HEALTH_PROGRESS_S) -> None:
    start = time.monotonic()
    deadline = start + timeout_s
    next_progress = start + progress_s
    while True:
        rc = handle.proc.poll()
        if rc is not None:
            raise RuntimeError(f"server exited with code {rc} before becoming healthy:\n{_log_tail(handle.log_path)}")
        try:
            if _get(f"{handle.base_url}/models") == 200:
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"server not healthy after {timeout_s:.0f}s:\n{_log_tail(handle.log_path)}")
        if now >= next_progress:
            last = (_log_tail(handle.log_path, 1) or "(no log yet)").strip()[:200]
            print(f"[setup_vllm] waiting for the server: {now - start:.0f}s; last log line: {last}", flush=True)
            next_progress = now + progress_s
        time.sleep(poll_s)


def gpu_memory() -> str:
    """`used / total MiB` per GPU from nvidia-smi, or why it is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({type(exc).__name__})"
    return "; ".join(line.strip() for line in out.splitlines() if line.strip())


def start_watchdog(handle: ServerHandle, poll_s: float = WATCH_POLL_S, gpu_s: float = WATCH_GPU_S) -> threading.Thread:
    """During the run: print the server's exit code and log tail the moment it dies (games would otherwise only
    show request failures), and GPU memory every `gpu_s` seconds. Stops at teardown."""
    def watch() -> None:
        next_gpu = time.monotonic() + gpu_s
        while not handle.torn_down:
            rc = handle.proc.poll()
            if rc is not None:
                if not handle.torn_down:
                    print(f"[setup_vllm] model server exited with code {rc} during the run:\n"
                          f"{_log_tail(handle.log_path)}", flush=True)
                return
            if time.monotonic() >= next_gpu:
                print(f"[setup_vllm] GPU memory: {gpu_memory()}", flush=True)
                next_gpu = time.monotonic() + gpu_s
            time.sleep(poll_s)

    thread = threading.Thread(target=watch, name="vllm-watchdog", daemon=True)
    thread.start()
    return thread


def smoke_chat(base_url: str, model: str = SERVED_MODEL, timeout_s: float = 300.0) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "stream": False,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    text = (data["choices"][0]["message"].get("content") or "").strip()
    print(f"[setup_vllm] smoke chat: {text[:80]!r}", flush=True)
    return text


def check_spec_acceptance(base_url: str, log_path: Path, model: str = SERVED_MODEL,
                          timeout_s: float = SPEC_CHECK_TIMEOUT_S, minimum: float = SPEC_MIN_ACCEPTANCE) -> float:
    """A predictable generation (counting), then vLLM's logged mean acceptance length must reach `minimum`: a draft
    head that did not load (random weights) accepts ~1.0 tokens per step and only slows the server down."""
    start = log_path.stat().st_size if log_path.is_file() else 0
    body = {"model": model, "stream": False, "max_tokens": 240, "temperature": 0.0,
            "messages": [{"role": "user", "content": "Count from 1 to 60, separated by commas. Output only the numbers."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        json.loads(r.read())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with open(log_path, "rb") as f:
            f.seek(start)
            found = re.findall(rb"Mean acceptance length: ([0-9.]+)", f.read())
        if found:
            value = float(found[-1])
            print(f"[setup_vllm] speculative decoding: mean acceptance length {value:.2f}", flush=True)
            if value < minimum:
                raise RuntimeError(f"MTP mean acceptance length {value:.2f} < {minimum}: the draft head did not load "
                                   f"or does not match the model\n{_log_tail(log_path)}")
            return value
        time.sleep(2.0)
    raise RuntimeError(f"no 'Mean acceptance length' line in the server log within {timeout_s:.0f}s of the "
                       f"speculative-decoding check\n{_log_tail(log_path)}")


def launch(cmd: list[str], env: dict[str, str], log_path: Path, pid_path: Path,
           site_packages: Path | None, base_url: str) -> ServerHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    pid_path.write_text(f"{proc.pid}\n")
    handle = ServerHandle(proc=proc, log_path=log_path, pid_path=pid_path, site_packages=site_packages, base_url=base_url)
    atexit.register(teardown, handle)
    return handle


def group_members(pgid: int) -> list[int]:
    """Live (non-zombie) pids in process group `pgid`, read from /proc."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()  # after "pid (comm) ": state ppid pgrp ...
        if int(fields[2]) == pgid and fields[0] != "Z":
            out.append(int(d.name))
    return out


def teardown(handle: ServerHandle, grace_s: float = KILL_GRACE_S, raise_errors: bool = True) -> None:
    """Stop the server (see `_stop`). With raise_errors=False (submission mode, where the scorecard is
    already closed and an exception would fail the notebook) an error is printed with its traceback instead."""
    try:
        _stop(handle, grace_s)
    except Exception as exc:  # noqa: BLE001 - reported, and re-raised unless the caller asked otherwise
        if raise_errors:
            raise
        print(f"[setup_vllm] teardown error after the run: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()


def _stop(handle: ServerHandle, grace_s: float) -> None:
    """SIGTERM the server's process group, SIGKILL after `grace_s`, then remove site-packages."""
    if handle.torn_down:
        return
    handle.torn_down = True
    pgid = handle.proc.pid  # start_new_session: the server leads its own group
    if group_members(pgid):
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while group_members(pgid) and time.monotonic() < deadline:
            handle.proc.poll()
            time.sleep(0.2)
        if group_members(pgid):
            os.killpg(pgid, signal.SIGKILL)
    handle.proc.wait()
    deadline = time.monotonic() + 10.0
    while group_members(pgid):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"server process group {pgid} still alive after SIGKILL: {group_members(pgid)}")
        time.sleep(0.1)
    handle.pid_path.unlink(missing_ok=True)
    if handle.site_packages is not None and handle.site_packages.exists():
        shutil.rmtree(handle.site_packages)
    print(f"[setup_vllm] server stopped (pid {pgid})", flush=True)


def export_env(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = str(v)


def start(*, model_dir: Path | None, wheelhouse: Path | None, working: Path, agent_env: dict[str, str],
          server_cmd: list[str] | None = None, host: str = HOST, port: int = PORT,
          check_gpu_first: bool = True, health_timeout_s: float = HEALTH_TIMEOUT_S,
          base_env: dict[str, str] | None = None, mtp_dir: Path | None = None) -> ServerHandle:
    """Bring the server up and export the agent env.

    `server_cmd` replaces the vLLM command (a stub server for local tests); with it,
    the GPU check and wheelhouse install are skipped because there is nothing to run them for.
    """
    t0 = time.monotonic()
    site_packages: Path | None = None
    spec = spec_tokens(agent_env)
    if spec and server_cmd is None and mtp_dir is None:
        raise ValueError(f"{SPEC_ENV}={spec} needs the MTP head dataset (paths.MTP_REF)")
    if server_cmd is None:
        if check_gpu_first:
            print(f"[setup_vllm] GPU: {check_gpu()}", flush=True)
        if model_dir is None or wheelhouse is None:
            raise ValueError("model_dir and wheelhouse are required to launch vLLM")
        model_dir = stage_model(model_dir)
        if spec:
            add_mtp_head(model_dir, mtp_dir)
        site_packages = working / "vllm-site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        fresh = install_wheelhouse(wheelhouse, site_packages)
        print(f"[setup_vllm] wheelhouse {'installed' if fresh else 'reused'} at {site_packages}", flush=True)
        cmd = vllm_command(model_dir, host, port, spec)
    else:
        cmd = list(server_cmd)
    base_url = f"http://{host}:{port}/v1"
    handle = launch(cmd, server_env(site_packages, base_env), working / "vllm-openai-server.log",
                    working / "vllm-openai-server.pid", site_packages, base_url)
    try:
        wait_healthy(handle, timeout_s=health_timeout_s)
        smoke_chat(base_url)
        if spec and server_cmd is None:
            check_spec_acceptance(base_url, handle.log_path)
    except BaseException:
        teardown(handle)
        raise
    handle.ready_s = time.monotonic() - t0
    env = {"LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url, "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL}
    env.update({k: str(v) for k, v in agent_env.items()})
    export_env(env)
    print(f"[setup_vllm] ready in {handle.ready_s:.0f}s at {base_url}", flush=True)
    return handle
