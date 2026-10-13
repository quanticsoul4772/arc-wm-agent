"""T5 tests: Kaggle builder, input search, vLLM setup/teardown, gateway simulator, notebook end to end.

Linux only (process groups, /proc). The end-to-end tests need the probe games'
environment files at $WMAGENT_TEST_ENV_FILES (default /env, as mounted by the
Docker test command) and install pip, pandas and pyarrow into the test venv.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from wmagent.kaggle import build, notebook, paths, setup_vllm
from wmagent.kaggle.gateway_sim import GatewaySim, prepare_env_dir

REPO = Path(__file__).resolve().parents[1]
ENV_FILES = Path(os.environ.get("WMAGENT_TEST_ENV_FILES", "/env"))
PROBE_GAMES = build.PROBE_GAMES


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------- builder


def dry_run(out: Path, *extra: str) -> tuple[Path, Path]:
    build.main(["--out", str(out), "--label", "t5", "--dry-run", *extra])
    return out / "kaggle" / "source-dataset", out / "kaggle" / "kernel"


def test_builder_dry_run_kernel_metadata(tmp_path):
    _, kernel = dry_run(tmp_path)
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert meta["id"] == "quanticsoul4772/wm-run-t5"
    assert meta["code_file"] == "run.ipynb" and (kernel / "run.ipynb").is_file()
    assert meta["language"] == "python" and meta["kernel_type"] == "notebook"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False and meta["is_private"] is True
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert meta["dataset_sources"] == [
        "quanticsoul4772/wm-src-t5",
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
    ]
    assert meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["docker_image"] == (
        "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")


def test_builder_dry_run_bundle(tmp_path):
    src, kernel = dry_run(tmp_path, "--n-passes", "1", "--concurrency", "6", "--probe-actions", "5")
    ds = json.loads((src / "dataset-metadata.json").read_text())
    assert ds["id"] == "quanticsoul4772/wm-src-t5"
    for f in ("setup_vllm.py", "run_config.json", "agent_env.json", paths.BUNDLE_MARKER, "src/wmagent/__init__.py",
              "src/wmagent/world_model_gate.py", "src/wmagent/kaggle/setup_vllm.py", "src/wmagent/kaggle/paths.py"):
        assert (src / f).is_file(), f
    assert not list(src.rglob("__pycache__"))
    marker = json.loads((src / paths.BUNDLE_MARKER).read_text())
    assert marker["slug"] == "wm-src-t5"
    assert set(marker["files"]) == {p.relative_to(src).as_posix() for p in src.rglob("*")
                                    if p.is_file() and p.name != paths.BUNDLE_MARKER}
    cfg = json.loads((src / "run_config.json").read_text())
    assert cfg["game_ids"] == PROBE_GAMES and cfg["n_passes"] == 1 and cfg["concurrency"] == 6
    assert cfg["max_runtime_s"] == 32400 and cfg["analyzer_timeout_s"] == 900 and cfg["cancel_drain_timeout_s"] == 120
    assert cfg["max_actions_per_game"] is None and cfg["max_runtime_s_per_game"] == 2700
    from wmagent.runner import RunConfig
    assert RunConfig.from_dict(cfg).n_passes == 1  # the runner (T1) accepts the bundle's config as is
    env = json.loads((src / "agent_env.json").read_text())
    assert env["PROBE_ACTIONS"] == "5" and env["PROBE_ENFORCE"] == "1" and env["PROBE_MAX_REFUSALS"] == "8"
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768" and env["LOCAL_ANALYZER_YIELD_SECONDS"] == "60"
    nb = json.loads((kernel / "run.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    script = notebook.as_script(nb)
    assert 'BUNDLE_SLUG = "wm-src-t5"' in script
    compile(script, "run.ipynb", "exec")
    assert "%" not in "".join(l[:1] for l in script.splitlines()) and "!pip" not in script  # no IPython magics


def test_push_retries_on_session_cap(tmp_path):
    calls, sleeps = [], []
    pushes = iter([
        (1, "403 - Maximum batch GPU session count of 1 reached"),
        (0, "Kernel push error: you have reached the limit of concurrent sessions"),
        (0, "Kernel version 3 successfully pushed."),
    ])

    def fake_run(cmd, **kw):
        calls.append(cmd[1:3])
        if cmd[1:3] == ["datasets", "status"]:
            return subprocess.CompletedProcess(cmd, 0, "ready", "")
        if cmd[1:3] == ["kernels", "push"]:
            rc, out = next(pushes)
            return subprocess.CompletedProcess(cmd, rc, out, "")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    build.push(tmp_path, tmp_path, "o/s", run=fake_run, sleep=sleeps.append, retry_wait_s=7)
    assert ["datasets", "version"] in calls and ["datasets", "create"] not in calls
    assert calls.count(["kernels", "push"]) == 3 and sleeps == [7, 7]


def test_push_fails_on_other_error(tmp_path):
    def fake_run(cmd, **kw):
        if cmd[1:3] == ["datasets", "status"]:
            return subprocess.CompletedProcess(cmd, 1, "", "404 Not Found") if not fake_run.created else \
                subprocess.CompletedProcess(cmd, 0, "ready", "")
        if cmd[1:3] == ["datasets", "create"]:
            fake_run.created = True
        if cmd[1:3] == ["kernels", "push"]:
            return subprocess.CompletedProcess(cmd, 1, "", "400 - invalid dataset source")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    fake_run.created = False
    with pytest.raises(RuntimeError, match="kernel push failed"):
        build.push(tmp_path, tmp_path, "o/s", run=fake_run, sleep=lambda s: None)
    assert fake_run.created


# ---------------------------------------------------------------- input search


def make_inputs(root: Path, *, layout: str, env_src: Path | None = None) -> dict[str, Path]:
    """Fake /kaggle/input in one of several mount layouts."""
    if layout == "documented":
        comp = root / "competitions" / paths.COMPETITION_SLUG
        wh = root / "datasets" / "driessmit1" / "arc3-vllm-h100-wheelhouse-v3"
        model = root / "datasets" / "driessmit1" / "vrfai-qwen3-6-27b-fp8-hf-snapshot"
    elif layout == "flat":
        comp = root / paths.COMPETITION_SLUG
        wh = root / "arc3-vllm-h100-wheelhouse-v3"
        model = root / "vrfai-qwen3-6-27b-fp8-hf-snapshot"
    else:  # moved: unknown parents and nested snapshot dirs
        comp = root / "mnt" / "c0" / "data"
        wh = root / "x" / "wheels-v3"
        model = root / "datasets" / "someone" / "snap" / "Qwen3.6-27B-FP8"
    (comp / "arc_agi_3_wheels").mkdir(parents=True)
    if env_src is not None:
        prepare_env_dir(env_src, comp / "environment_files")
    else:
        (comp / "environment_files").mkdir()
    wh.mkdir(parents=True)
    (wh / "requirements.lock").write_text("torch==2.10.0\nvllm==0.19.0\n")
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"")
    # decoys that must not be picked
    (root / "decoy" / "requirements.lock").parent.mkdir(parents=True)
    (root / "decoy" / "requirements.lock").write_text("numpy==2\n")
    return {"comp": comp, "wh": wh, "model": model}


@pytest.mark.parametrize("layout", ["documented", "flat", "moved"])
def test_inputs_found_by_search(tmp_path, layout):
    want = make_inputs(tmp_path, layout=layout)
    assert paths.find_competition(tmp_path) == want["comp"]
    assert paths.find_wheelhouse(tmp_path) == want["wh"]
    assert paths.find_model(tmp_path) == want["model"]


def test_bundle_found_by_marker_prefers_slug(tmp_path):
    for d in ("datasets/o/other-bundle", "datasets/o/wm-src-x"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / paths.BUNDLE_MARKER).write_text("{}")
    assert paths.find_bundle("wm-src-x", tmp_path) == tmp_path / "datasets/o/wm-src-x"
    with pytest.raises(FileNotFoundError):
        paths.find_model(tmp_path)


# ---------------------------------------------------------------- setup_vllm


def test_vllm_command_flags(tmp_path):
    cmd = setup_vllm.vllm_command(tmp_path / "m")
    s = " ".join(cmd)
    assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    for flag in ("--model " + str(tmp_path / "m"), "--served-model-name vrfai/Qwen3.6-27B-FP8", "--host 127.0.0.1",
                 "--port 1234", "--tensor-parallel-size 1", "--enable-auto-tool-choice",
                 "--tool-call-parser qwen3_coder", "--reasoning-parser qwen3", "--generation-config vllm",
                 "--enable-prefix-caching", "--max-model-len 65536"):
        assert flag in s, flag
    i = cmd.index("--default-chat-template-kwargs")
    assert json.loads(cmd[i + 1]) == {"preserve_thinking": True}
    env = setup_vllm.server_env(tmp_path / "sp")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path / "sp")
    assert env["USE_TF"] == "0" and env["VLLM_NO_USAGE_STATS"] == "1" and env["TRANSFORMERS_NO_TORCHVISION"] == "1"


def test_gpu_check(monkeypatch):
    def fake(out):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, out, "")
    monkeypatch.setattr(setup_vllm.subprocess, "run", fake("NVIDIA RTX PRO 6000 Blackwell Server Edition\n"))
    assert setup_vllm.check_gpu()
    for bad in ("NVIDIA H100 80GB HBM3\n", "NVIDIA RTX PRO 6000\nNVIDIA RTX PRO 6000\n", ""):
        monkeypatch.setattr(setup_vllm.subprocess, "run", fake(bad))
        with pytest.raises(RuntimeError):
            setup_vllm.check_gpu()


def test_wheelhouse_install_and_stamp_reuse(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(setup_vllm.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    target = tmp_path / "vllm-site-packages"
    target.mkdir()
    assert setup_vllm.install_wheelhouse(tmp_path / "wh", target) is True
    c = cmds[0]
    assert c[1:4] == ["-m", "pip", "install"] and "--no-index" in c and c[c.index("--target") + 1] == str(target)
    assert c[c.index("-r") + 1] == str(tmp_path / "wh" / "requirements.lock")
    for f in ("--only-binary", ":all:", "--no-compile", "--ignore-installed", "--upgrade"):
        assert f in c
    assert setup_vllm.install_wheelhouse(tmp_path / "wh", target) is False and len(cmds) == 1
    (target / setup_vllm.STAMP_FILE).write_text("vllm==0.18.0\n")
    assert setup_vllm.install_wheelhouse(tmp_path / "wh", target) is True


def test_start_with_stub_exports_env_and_teardown_kills(tmp_path, monkeypatch):
    port = free_port()
    for k in ("LOCAL_ANALYZER_BASE_URL", "PROBE_ACTIONS"):
        monkeypatch.delenv(k, raising=False)
    h = setup_vllm.start(model_dir=None, wheelhouse=None, working=tmp_path, agent_env={"PROBE_ACTIONS": "8"},
                         server_cmd=[sys.executable, "-m", "wmagent.kaggle.stub_openai", "--port", str(port)],
                         port=port, health_timeout_s=60)
    try:
        assert os.environ["LOCAL_ANALYZER_BASE_URL"] == f"http://127.0.0.1:{port}/v1"
        assert os.environ["OPENAI_BASE_URL"] == os.environ["LOCAL_ANALYZER_BASE_URL"]
        assert os.environ["LOCAL_ANALYZER_MODEL_ID"] == "vrfai/Qwen3.6-27B-FP8"
        assert os.environ["PROBE_ACTIONS"] == "8"
        assert int((tmp_path / "vllm-openai-server.pid").read_text()) == h.proc.pid
        assert port_open(port)
    finally:
        setup_vllm.teardown(h)
    assert setup_vllm.group_members(h.proc.pid) == [] and not port_open(port)
    assert not (tmp_path / "vllm-openai-server.pid").exists()


def test_teardown_escalates_to_sigkill_and_kills_children(tmp_path):
    code = textwrap.dedent("""
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"])
        print("up", flush=True)
        time.sleep(600)
    """)
    sp = tmp_path / "vllm-site-packages"
    sp.mkdir()
    h = setup_vllm.launch([sys.executable, "-c", code], dict(os.environ), tmp_path / "log", tmp_path / "pid", sp, "x")
    deadline = time.monotonic() + 10
    while len(setup_vllm.group_members(h.proc.pid)) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert len(setup_vllm.group_members(h.proc.pid)) == 2
    t0 = time.monotonic()
    setup_vllm.teardown(h, grace_s=1.0)
    assert time.monotonic() - t0 < 10
    assert setup_vllm.group_members(h.proc.pid) == [] and h.proc.returncode == -signal.SIGKILL
    assert not sp.exists()


def test_start_fails_fast_when_server_dies(tmp_path):
    with pytest.raises(RuntimeError, match="exited with code 3"):
        setup_vllm.start(model_dir=None, wheelhouse=None, working=tmp_path, agent_env={},
                         server_cmd=[sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
                         port=free_port(), health_timeout_s=60)


# ---------------------------------------------------------------- gateway simulator


def require_env_files() -> Path:
    assert (ENV_FILES / "bp35").is_dir(), f"probe environment files missing at {ENV_FILES} (set WMAGENT_TEST_ENV_FILES)"
    return ENV_FILES


@pytest.fixture
def sim(tmp_path):
    env_dir = tmp_path / "served"
    prepare_env_dir(require_env_files(), env_dir)
    s = GatewaySim(env_dir, tmp_path / "sim-out")
    url = s.serve_background()
    yield s, url
    s.shutdown()


def competition_arcade(url: str, monkeypatch, api_key: str = "test-key-123"):
    import logging

    from arc_agi import Arcade, OperationMode
    monkeypatch.setenv("OPERATION_MODE", "competition")
    if api_key:
        monkeypatch.setenv("ARC_API_KEY", api_key)
    else:
        monkeypatch.delenv("ARC_API_KEY", raising=False)
    log = logging.getLogger("t5-client")
    log.setLevel(logging.CRITICAL)
    return Arcade(operation_mode=OperationMode.COMPETITION, arc_base_url=url, environments_dir="", logger=log)


def sim_status(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/sim/status", timeout=10) as r:
        return json.loads(r.read())


def test_sim_competition_one_scorecard_each_game_once(sim, monkeypatch, tmp_path):
    from arcengine import GameAction, GameState
    _, url = sim
    arcade = competition_arcade(url, monkeypatch)
    ids = sorted(e.game_id for e in arcade.available_environments)
    assert ids == sorted(PROBE_GAMES)
    card = arcade.create_scorecard()
    import requests
    with pytest.raises(requests.HTTPError):
        arcade.create_scorecard()  # competition mode: a second scorecard is refused
    envs = {}
    for gid in ids:
        env = arcade.make(gid, scorecard_id=card)
        assert env is not None and env.observation_space.state == GameState.NOT_FINISHED
        assert env.environment_info.game_id == gid
        envs[gid] = env
    assert arcade.make(ids[0], scorecard_id=card).observation_space is None  # each id once
    for gid, env in envs.items():
        obs = env.observation_space
        acts = [GameAction.from_id(a) for a in obs.available_actions]
        a = next((x for x in acts if x.is_simple()), acts[0])
        data = {} if a.is_simple() else {"x": 32, "y": 32}
        obs = env.step(a, data=data)
        assert obs is not None and obs.frame
    sc = arcade.close_scorecard(card)
    assert sc is not None
    st = sim_status(url)
    assert st["scorecards_opened"] == 1 and st["open_rejected"] == 1 and st["scorecards_closed"] == [card]
    assert st["makes"] == {gid: 1 for gid in ids}
    closed = json.loads((tmp_path / "sim-out" / "scorecard_closed.json").read_text())
    assert closed["card_id"] == card and "api_key" not in closed


def test_sim_anonkey_route(sim, monkeypatch):
    _, url = sim
    with urllib.request.urlopen(f"{url}/api/games/anonkey", timeout=10) as r:
        assert json.loads(r.read())["api_key"].startswith("anon-")
    with urllib.request.urlopen(f"{url}/api/games/bp35", timeout=10) as r:
        assert json.loads(r.read())["game_id"] == "bp35-0a0ad940"
    arcade = competition_arcade(url, monkeypatch, api_key="")
    assert arcade.arc_api_key.startswith("anon-") and len(arcade.available_environments) == 6


def test_sim_clone_to(tmp_path, monkeypatch):
    env_dir = tmp_path / "served"
    ids = prepare_env_dir(require_env_files(), env_dir, clone_to=14)
    assert len(ids) == len(set(ids)) == 14 and len({i.split("-")[0] for i in ids}) == 14
    s = GatewaySim(env_dir, tmp_path / "sim-out")
    url = s.serve_background()
    try:
        arcade = competition_arcade(url, monkeypatch)
        assert len(arcade.available_environments) == 14
        card = arcade.create_scorecard()
        clone = next(i for i in ids if i.startswith("k0"))
        env = arcade.make(clone, scorecard_id=card)
        assert env is not None and env.environment_info.game_id == clone and env.observation_space.frame
        arcade.close_scorecard(card)
        assert sim_status(url)["makes"][clone] == 1
    finally:
        s.shutdown()


# ---------------------------------------------------------------- notebook end to end

FAKE_RUNNER = '''
"""Stand-in for wmagent.runner.run used by the notebook tests: one scorecard, each game made once,
a few legal actions per game (chosen after one chat call to the analyzer server), then close."""
import json, logging, os, urllib.request
from pathlib import Path

def _chat():
    base = os.environ["LOCAL_ANALYZER_BASE_URL"]
    body = {"model": os.environ["LOCAL_ANALYZER_MODEL_ID"], "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "python", "parameters": {}}}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["tool_calls"][0]["function"]["name"]

def run(config, *, output_dir, submission, notebook_start_epoch):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    log = logging.getLogger("fake-runner"); log.setLevel(logging.WARNING)
    if config["game_ids"] == "competition":
        arcade = Arcade(operation_mode=OperationMode.COMPETITION, arc_base_url=os.environ["ARC_BASE_URL"],
                        environments_dir="", logger=log)
        game_ids = [e.game_id for e in arcade.available_environments]
    else:
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=config["environments_dir"], logger=log)
        game_ids = config["game_ids"]
    card = arcade.create_scorecard()
    played = {}
    for gid in game_ids:
        env = arcade.make(gid, scorecard_id=card)
        assert _chat() == "python"
        n = 0
        for _ in range(3):
            obs = env.observation_space
            acts = [GameAction.from_id(a) for a in obs.available_actions]
            a = next((x for x in acts if x.is_simple()), acts[0])
            env.step(a, data={} if a.is_simple() else {"x": 32, "y": 32})
            n += 1
        played[gid] = n
        print(f"[finished] {gid} actions={n}", flush=True)
    arcade.close_scorecard(card)
    Path(output_dir, "fake_runner.json").write_text(json.dumps(
        {"config": config, "submission": submission, "start": notebook_start_epoch, "played": played,
         "wmagent_file": __import__("wmagent").__file__,
         "env": {k: os.environ.get(k) for k in ("PROBE_ACTIONS", "LOCAL_ANALYZER_MODEL_ID", "ONLY_RESET_LEVELS",
                                                  "OPERATION_MODE", "LIBRARY_PATH")}}))
'''


@pytest.fixture(scope="session")
def notebook_env_python():
    """The test venv with pip (the notebook pip-installs arc-agi) and pandas/pyarrow (offline submission file)."""
    py = sys.executable
    if subprocess.run([py, "-m", "pip", "--version"], check=False, capture_output=True).returncode != 0:
        subprocess.run([py, "-m", "ensurepip"], check=True)
    if subprocess.run([py, "-c", "import pandas, pyarrow"], check=False, capture_output=True).returncode != 0:
        subprocess.run(["uv", "pip", "install", "-q", "--python", py, "pandas", "pyarrow"], check=True)
    return py


@pytest.fixture
def notebook_python(notebook_env_python):
    """Per test: remove the .pth file the notebook writes into the interpreter's site-packages."""
    import sysconfig
    pth = Path(sysconfig.get_paths()["purelib"], "wmagent-bundle.pth")
    yield notebook_env_python
    pth.unlink(missing_ok=True)


def stage_notebook_run(tmp_path: Path, *build_args: str, fake_runner: bool = True
                       ) -> tuple[Path, dict[str, str], Path, Path]:
    run_dir = tmp_path / "run"
    src, kernel = dry_run(run_dir, "--n-passes", "1", "--concurrency", "6", *build_args)
    inp, work = tmp_path / "input", tmp_path / "working"
    make_inputs(inp, layout="moved", env_src=require_env_files())
    bundle = inp / "datasets" / "quanticsoul4772" / "wm-src-t5"
    bundle.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(src, bundle)
    runner_dir = tmp_path / "fake"
    runner_dir.mkdir()
    (runner_dir / "fake_runner.py").write_text(FAKE_RUNNER)
    script = tmp_path / "run_notebook.py"
    script.write_text(notebook.as_script(json.loads((kernel / "run.ipynb").read_text())))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "OPERATION_MODE", "KAGGLE_",
                                                                    "LOCAL_ANALYZER", "PROBE_", "ONLY_RESET"))}
    env.update({
        "WMAGENT_KAGGLE_INPUT": str(inp), "WMAGENT_KAGGLE_WORKING": str(work), "WMAGENT_STUB_LLM": "1",
        "WMAGENT_LLM_PORT": str(free_port()), "PYTHONPATH": str(runner_dir),
    })
    if fake_runner:
        env["WMAGENT_RUNNER"] = "fake_runner:run"
    return script, env, work, bundle


def run_notebook(py: str, script: Path, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    p = subprocess.run([py, str(script)], check=False, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, f"notebook failed\nSTDOUT:\n{p.stdout[-4000:]}\nSTDERR:\n{p.stderr[-4000:]}"
    return p


def assert_server_gone(work: Path, port: int) -> None:
    assert not (work / "vllm-openai-server.pid").exists()
    assert not port_open(port)
    assert procs_matching("wmagent.kaggle.stub_openai") == []


def procs_matching(needle: str) -> list[str]:
    """Command lines of live processes containing `needle` (no pgrep in the image)."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != os.getpid():
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
                state = (d / "stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if needle in cmd and state != "Z":
                out.append(cmd)
    return out


def test_notebook_offline_end_to_end(tmp_path, notebook_python):
    script, env, work, bundle = stage_notebook_run(tmp_path)
    p = run_notebook(notebook_python, script, env, tmp_path)
    assert f"bundle={bundle}" in p.stdout
    res = json.loads((work / "fake_runner.json").read_text())
    cfg = res["config"]
    assert res["submission"] is False and cfg["n_passes"] == 1
    assert cfg["environments_dir"].endswith("mnt/c0/data/environment_files")
    assert cfg["game_ids"] == PROBE_GAMES and set(res["played"]) == set(PROBE_GAMES)
    assert res["env"]["PROBE_ACTIONS"] == "8" and res["env"]["ONLY_RESET_LEVELS"] == "true"
    assert res["env"]["LIBRARY_PATH"].split(os.pathsep)[0] == "/usr/local/nvidia/lib64"
    import pandas as pd
    df = pd.read_parquet(work / "submission.parquet")
    assert list(df.columns) == ["row_id", "game_id", "end_of_game", "score"]
    assert df.to_dict("records") == [{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1}]
    assert "[finished] bp35-0a0ad940" in (work / "stdout.log").read_text()
    assert (work / "vllm-openai-server.log").read_text().count("POST /v1/chat/completions") == 1 + len(PROBE_GAMES)
    import sysconfig
    assert Path(sysconfig.get_paths()["purelib"], "wmagent-bundle.pth").read_text().strip() == str(bundle / "src")
    assert res["wmagent_file"] == str(bundle / "src" / "wmagent" / "__init__.py")  # the bundle copy, not the checkout
    assert_server_gone(work, int(env["WMAGENT_LLM_PORT"]))


def test_notebook_competition_end_to_end(tmp_path, notebook_python):
    script, env, work, _ = stage_notebook_run(tmp_path)
    port = free_port()
    sim_out = tmp_path / "sim"
    sim_env = dict(env, PYTHONPATH=str(REPO))
    sim_proc = subprocess.Popen(
        [notebook_python, "-m", "wmagent.kaggle.gateway_sim", "--env-dir", str(require_env_files()),
         "--out-dir", str(sim_out), "--port", str(port), "--clone-to", "8"],
        env=sim_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        env.update({"KAGGLE_IS_COMPETITION_RERUN": "true", "ARC_BASE_URL": f"http://127.0.0.1:{port}/"})
        run_notebook(notebook_python, script, env, tmp_path)  # the notebook itself waits for the gateway
        st = sim_status(f"http://127.0.0.1:{port}")
    finally:
        sim_proc.terminate()
        sim_proc.wait(timeout=30)
    res = json.loads((work / "fake_runner.json").read_text())
    cfg = res["config"]
    assert res["submission"] is True and cfg["game_ids"] == "competition" and cfg["n_passes"] == 1
    assert cfg["environments_dir"] is None and len(res["played"]) == 8
    assert res["env"]["OPERATION_MODE"] == "competition"
    assert st["scorecards_opened"] == 1 and len(st["scorecards_closed"]) == 1
    assert st["makes"] == {gid: 1 for gid in res["played"]}
    assert (sim_out / "scorecard_closed.json").is_file()
    # submission mode: the notebook writes no submission file, no stdout tee, no benchmark
    assert not (work / "submission.parquet").exists() and not (work / "stdout.log").exists()
    assert_server_gone(work, int(env["WMAGENT_LLM_PORT"]))


def test_notebook_tears_down_when_run_fails(tmp_path, notebook_python):
    script, env, work, _ = stage_notebook_run(tmp_path)
    (tmp_path / "fake" / "fake_runner.py").write_text("def run(config, **kw):\n    raise RuntimeError('runner blew up')\n")
    p = subprocess.run([notebook_python, str(script)], check=False, env=env, cwd=tmp_path, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode != 0 and "runner blew up" in p.stderr
    assert "server stopped" in p.stdout
    assert_server_gone(work, int(env["WMAGENT_LLM_PORT"]))


# ---------------------------------------------------------------- notebook with the real runner, session and agent


def assert_no_crashes(games: list[dict]) -> None:
    crashed = {g["game_id"]: g["solver_note"] for g in games if g["state"] == "crashed"}
    assert not crashed, f"crashed games: {crashed}"


def test_notebook_offline_real_stack(tmp_path, notebook_python):
    script, env, work, _ = stage_notebook_run(tmp_path, "--max-actions-per-game", "3", "--per-game-cap-s", "300",
                                              fake_runner=False)
    p = run_notebook(notebook_python, script, env, tmp_path)
    bench = json.loads((work / "benchmark.json").read_text())
    runs = bench["game_runs"]
    assert bench["label"] == "t5" and sorted(r["game_id"] for r in runs) == sorted(PROBE_GAMES)
    assert_no_crashes(runs)
    assert all(1 <= len(r["history"]) <= 3 for r in runs), [len(r["history"]) for r in runs]
    log = (work / "stdout.log").read_text()
    assert log.count("[finished]") == len(PROBE_GAMES), p.stdout[-3000:]
    assert len(list((work / "transcripts").glob("*.txt"))) == len(PROBE_GAMES)
    assert (work / "submission.parquet").is_file()
    assert_server_gone(work, int(env["WMAGENT_LLM_PORT"]))


def test_notebook_competition_real_stack(tmp_path, notebook_python):
    script, env, work, _ = stage_notebook_run(tmp_path, "--max-actions-per-game", "2", "--per-game-cap-s", "300",
                                              fake_runner=False)
    port = free_port()
    sim_out = tmp_path / "sim"
    sim_proc = subprocess.Popen(
        [notebook_python, "-m", "wmagent.kaggle.gateway_sim", "--env-dir", str(require_env_files()),
         "--out-dir", str(sim_out), "--port", str(port), "--clone-to", "8"],
        env=dict(env, PYTHONPATH=str(REPO)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        env.update({"KAGGLE_IS_COMPETITION_RERUN": "1", "ARC_BASE_URL": f"http://127.0.0.1:{port}/"})
        p = run_notebook(notebook_python, script, env, tmp_path)
        st = sim_status(f"http://127.0.0.1:{port}")
    finally:
        sim_proc.terminate()
        sim_proc.wait(timeout=30)
    assert st["scorecards_opened"] == 1 and st["open_rejected"] == 0 and len(st["scorecards_closed"]) == 1, st
    assert st["makes"] == {gid: 1 for gid in st["games"]} and len(st["games"]) == 8
    assert p.stdout.count("[finished]") == 8
    assert "state=crashed" not in p.stdout, p.stdout[-3000:]
    closed = json.loads((sim_out / "scorecard_closed.json").read_text())
    assert closed["card_id"] == st["scorecards_closed"][0]
    # submission mode: no benchmark.json, prompts, stdout tee or submission file from the notebook
    for name in ("benchmark.json", "prompts", "stdout.log", "submission.parquet"):
        assert not (work / name).exists(), name
    assert_server_gone(work, int(env["WMAGENT_LLM_PORT"]))
