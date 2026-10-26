"""The Kaggle notebook (`run.ipynb`), as plain-Python cells.

Cells use no IPython magics, so the notebook also runs as a script (the local
end-to-end test concatenates the code cells). Behavior follows spec 2.6:

1. start time, submission flag, LIBRARY_PATH, ONLY_RESET_LEVELS;
2. locate inputs by search (bundle by marker file; competition, wheelhouse, model);
3. install arc-agi from the competition wheels;
4. start the vLLM server via the bundle's setup_vllm.py;
5. load run_config.json, prepare offline or competition mode, run, write the
   offline dummy submission, and always tear the server down.

Local-test overrides (unset on Kaggle): WMAGENT_KAGGLE_INPUT / WMAGENT_KAGGLE_WORKING
(roots), WMAGENT_STUB_LLM=1 (stub server instead of vLLM), WMAGENT_LLM_PORT,
WMAGENT_RUNNER (module:function), and ARC_BASE_URL (gateway address).
"""
from __future__ import annotations

SLUG_TOKEN = "__WMAGENT_BUNDLE_SLUG__"

CELL_START = '''\
import json, os, sys, time
from pathlib import Path

NOTEBOOK_START_EPOCH = time.time()
SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes", "y", "on"}
os.environ["LIBRARY_PATH"] = os.pathsep.join(p for p in ("/usr/local/nvidia/lib64", os.environ.get("LIBRARY_PATH", "")) if p)
os.environ["ONLY_RESET_LEVELS"] = "true"
INPUT = Path(os.environ.get("WMAGENT_KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("WMAGENT_KAGGLE_WORKING", "/kaggle/working"))
WORKING.mkdir(parents=True, exist_ok=True)
# Kaggle's cwd. It matters: arc_agi scans ENVIRONMENTS_DIR="" as the cwd, so in competition mode any
# metadata.json below the cwd would be listed next to the gateway's games.
os.chdir(WORKING)
BUNDLE_SLUG = "__WMAGENT_BUNDLE_SLUG__"
STUB_LLM = os.environ.get("WMAGENT_STUB_LLM", "") == "1"
print(f"[notebook] start={NOTEBOOK_START_EPOCH:.0f} submission={SUBMISSION} stub_llm={STUB_LLM}", flush=True)
'''

CELL_LOCATE = '''\
import sysconfig

def _find_bundle(root, slug, max_depth=4):
    hits, frontier = [], [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            if (d / "wmagent-bundle.json").is_file():
                hits.append(d)
            try:
                nxt.extend(sorted(c for c in d.iterdir() if c.is_dir() and c.name not in ("environment_files", "src")))
            except OSError:
                pass
        frontier = nxt
    if not hits:
        raise FileNotFoundError(f"code bundle (wmagent-bundle.json) not found under {root}")
    hits.sort(key=lambda d: (d.name != slug, len(d.parts), str(d)))
    return hits[0]

BUNDLE = _find_bundle(INPUT, BUNDLE_SLUG)
BUNDLE_SRC = BUNDLE / "src"
sys.path.insert(0, str(BUNDLE_SRC))
Path(sysconfig.get_paths()["purelib"], "wmagent-bundle.pth").write_text(str(BUNDLE_SRC) + "\\n")

from wmagent.kaggle import paths as _paths
COMPETITION = _paths.find_competition(INPUT)
WHEELHOUSE = None if STUB_LLM else _paths.find_wheelhouse(INPUT)
MODEL_DIR = None if STUB_LLM else _paths.find_model(INPUT)
print(f"[notebook] bundle={BUNDLE} competition={COMPETITION} wheelhouse={WHEELHOUSE} model={MODEL_DIR}", flush=True)
'''

CELL_ARC_AGI = '''\
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links",
                str(COMPETITION / "arc_agi_3_wheels"), "arc-agi"], check=True)
'''

CELL_VLLM = '''\
import importlib.util
_spec = importlib.util.spec_from_file_location("setup_vllm", BUNDLE / "setup_vllm.py")
setup_vllm = importlib.util.module_from_spec(_spec)
sys.modules["setup_vllm"] = setup_vllm
_spec.loader.exec_module(setup_vllm)

AGENT_ENV = json.loads((BUNDLE / "agent_env.json").read_text())
LLM_PORT = int(os.environ.get("WMAGENT_LLM_PORT", setup_vllm.PORT))
SERVER = None
_SERVER_ERROR = []

def _start_server(base_env):
    global SERVER
    try:
        if STUB_LLM:
            time.sleep(float(os.environ.get("WMAGENT_STUB_DELAY_S", "0")))  # tests: a slow model load
        SERVER = setup_vllm.start(
            model_dir=MODEL_DIR, wheelhouse=WHEELHOUSE, working=WORKING, agent_env=AGENT_ENV,
            server_cmd=[sys.executable, "-m", "wmagent.kaggle.stub_openai", "--port", str(LLM_PORT)] if STUB_LLM else None,
            port=LLM_PORT, base_env=base_env,
        )
    except BaseException as exc:  # re-raised by _server_ready on the main thread
        _SERVER_ERROR.append(exc)

if SUBMISSION:
    # A competition rerun loads the model in the background while the games are made and kept
    # alive (Runner.run): a run with no move for 15 minutes is reported to end there.
    import threading
    _SERVER_THREAD = threading.Thread(target=_start_server, args=(dict(os.environ),), daemon=True)
    _SERVER_THREAD.start()
else:
    _SERVER_THREAD = None
    _start_server(dict(os.environ))
    if _SERVER_ERROR:
        raise _SERVER_ERROR[0]

def _server_ready():
    if _SERVER_THREAD is not None:
        _SERVER_THREAD.join()
    if _SERVER_ERROR:
        raise _SERVER_ERROR[0]
'''

CELL_RUN = '''\
import contextlib, importlib, urllib.error, urllib.request

COMPETITION_ENV = {
    "ARC_API_KEY": "test-key-123", "ARC_BASE_URL": "http://gateway:8001/", "SCHEME": "http", "HOST": "gateway",
    "PORT": "8001", "OPERATION_MODE": "competition", "ENVIRONMENTS_DIR": "", "RECORDINGS_DIR": str(WORKING / "server_recording"),
}

def _wait_gateway(base, timeout_s=600.0, poll_s=5.0):
    deadline = time.monotonic() + timeout_s
    req = urllib.request.Request(f"{base}/api/games", headers={"X-API-Key": os.environ["ARC_API_KEY"]})
    while True:
        status = None
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except OSError as e:
            print(f"[notebook] gateway not ready: {e}", flush=True)
        if status is not None and status < 500:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"gateway {base} not ready after {timeout_s:.0f}s (last status {status})")
        time.sleep(poll_s)

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()

try:
    config = json.loads((BUNDLE / "run_config.json").read_text())
    if SUBMISSION:
        for k, v in COMPETITION_ENV.items():
            os.environ.setdefault(k, v)
        _wait_gateway(os.environ["ARC_BASE_URL"].rstrip("/"))
        # The runner lists the gateway's games (error if none), makes each once, plays 1 pass and
        # closes the one shared scorecard before its submission deadline (spec 2.2, R3).
        config.update(game_ids="competition", n_passes=1, environments_dir=None)
    else:
        config.update(environments_dir=str(COMPETITION / "environment_files"))
    print(f"[notebook] games={config['game_ids']} passes={config['n_passes']}", flush=True)
    _mod, _fn = os.environ.get("WMAGENT_RUNNER", "wmagent.kaggle.entry:run").split(":")
    run = getattr(importlib.import_module(_mod), _fn)
    if SUBMISSION:
        run(config, output_dir=str(WORKING), submission=True, notebook_start_epoch=NOTEBOOK_START_EPOCH,
            ready=_server_ready)
    else:
        with open(WORKING / "stdout.log", "a", encoding="utf-8") as _log, contextlib.redirect_stdout(_Tee(sys.stdout, _log)):
            run(config, output_dir=str(WORKING), submission=False, notebook_start_epoch=NOTEBOOK_START_EPOCH)
        import pandas as pd
        pd.DataFrame([{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1}]).to_parquet(
            WORKING / "submission.parquet", index=False)
    print(f"[notebook] done in {time.time() - NOTEBOOK_START_EPOCH:.0f}s", flush=True)
finally:
    # In a submission the scorecard is closed by now; a teardown error must not fail the notebook.
    if _SERVER_THREAD is not None:
        _SERVER_THREAD.join(timeout=60)  # an early failure here must not leave a late launch running
    if SERVER is not None:
        setup_vllm.teardown(SERVER, raise_errors=not SUBMISSION)
'''

CODE_CELLS = [CELL_START, CELL_LOCATE, CELL_ARC_AGI, CELL_VLLM, CELL_RUN]
TITLE_MD = "# arc-wm-agent run\n\nOffline run or competition rerun of the world-model agent (spec 2.6)."


def _source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def render(bundle_slug: str) -> dict:
    """The notebook as an nbformat-4 dict, with the bundle slug filled in."""
    cells: list[dict] = [{"cell_type": "markdown", "id": "cell-0", "metadata": {}, "source": _source(TITLE_MD)}]
    for i, code in enumerate(CODE_CELLS, 1):
        cells.append({
            "cell_type": "code", "id": f"cell-{i}", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _source(code.replace(SLUG_TOKEN, bundle_slug)),
        })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def as_script(nb: dict) -> str:
    """Concatenate the code cells (for running the notebook outside Jupyter)."""
    return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
