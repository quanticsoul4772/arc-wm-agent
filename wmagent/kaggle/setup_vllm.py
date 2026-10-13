# ruff: noqa: EXE002 - the mode bit comes from the Windows bind mount, not the file
"""Install and start the local vLLM OpenAI server for a Kaggle run (stdlib only).

The builder copies this file to the code bundle root; the notebook loads it with
runpy, so it must not import the wmagent package. `start()` returns a handle;
`teardown(handle)` stops the server process group and removes the installed
site-packages.

Sequence (spec 2.6 step 4): GPU check, wheelhouse install (reused when the
stamp matches), server launch, /v1/models health poll, one smoke chat, and the
analyzer env exported into os.environ.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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
HEALTH_TIMEOUT_S = 900.0
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
    subprocess.run(cmd, check=True)
    stamp.write_text(STAMP + "\n")
    return True


def vllm_command(model_dir: Path, host: str = HOST, port: int = PORT) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir), "--served-model-name", SERVED_MODEL,
        "--host", host, "--port", str(port), "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--generation-config", "vllm", "--enable-prefix-caching",
        "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
        "--max-model-len", "65536",
    ]


def server_env(site_packages: Path | None) -> dict[str, str]:
    env = dict(os.environ)
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


def wait_healthy(handle: ServerHandle, timeout_s: float = HEALTH_TIMEOUT_S, poll_s: float = HEALTH_POLL_S) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        rc = handle.proc.poll()
        if rc is not None:
            raise RuntimeError(f"server exited with code {rc} before becoming healthy:\n{_log_tail(handle.log_path)}")
        try:
            if _get(f"{handle.base_url}/models") == 200:
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"server not healthy after {timeout_s:.0f}s:\n{_log_tail(handle.log_path)}")
        time.sleep(poll_s)


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


def teardown(handle: ServerHandle, grace_s: float = KILL_GRACE_S) -> None:
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
          check_gpu_first: bool = True, health_timeout_s: float = HEALTH_TIMEOUT_S) -> ServerHandle:
    """Bring the server up and export the agent env.

    `server_cmd` replaces the vLLM command (a stub server for local tests); with it,
    the GPU check and wheelhouse install are skipped because there is nothing to run them for.
    """
    t0 = time.monotonic()
    site_packages: Path | None = None
    if server_cmd is None:
        if check_gpu_first:
            print(f"[setup_vllm] GPU: {check_gpu()}", flush=True)
        if model_dir is None or wheelhouse is None:
            raise ValueError("model_dir and wheelhouse are required to launch vLLM")
        site_packages = working / "vllm-site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        fresh = install_wheelhouse(wheelhouse, site_packages)
        print(f"[setup_vllm] wheelhouse {'installed' if fresh else 'reused'} at {site_packages}", flush=True)
        cmd = vllm_command(model_dir, host, port)
    else:
        cmd = list(server_cmd)
    base_url = f"http://{host}:{port}/v1"
    handle = launch(cmd, server_env(site_packages), working / "vllm-openai-server.log",
                    working / "vllm-openai-server.pid", site_packages, base_url)
    try:
        wait_healthy(handle, timeout_s=health_timeout_s)
        smoke_chat(base_url)
    except BaseException:
        teardown(handle)
        raise
    handle.ready_s = time.monotonic() - t0
    env = {"LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url, "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL}
    env.update({k: str(v) for k, v in agent_env.items()})
    export_env(env)
    print(f"[setup_vllm] ready in {handle.ready_s:.0f}s at {base_url}", flush=True)
    return handle
