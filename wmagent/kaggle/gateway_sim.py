"""Local stand-in for the Kaggle competition gateway.

Serves arc_agi's own REST app (`arc_agi.server.create_app`) in competition mode
over an OFFLINE Arcade of local environment files, plus:

- `GET /api/games/anonkey` (the client asks for it when ARC_API_KEY is empty);
- `GET /sim/status`: scorecards opened and closed, makes per game id, rejected
  duplicate opens, and the closed scorecard's summary.

On scorecard close it writes `<out_dir>/scorecard_closed.json` (the scoring side
of a real rerun turns the closed scorecard into the result; this simulator does
not write a submission.parquet because that format is not known to us).

`--clone-to N` copies the games under new ids until N games exist (e.g. the 25
public games cloned to 110 ids, as the hidden set is sized), keeping each
clone's class name so the original game file loads.

    python -m wmagent.kaggle.gateway_sim --env-dir ENV --out-dir OUT --port 8001 [--clone-to 110]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from arc_agi import Arcade, OperationMode
from arc_agi.server import create_app
from flask import jsonify
from werkzeug.serving import make_server

# Client-side variables that would leak into the gateway's own Arcade: arc_agi lets
# OPERATION_MODE=competition override the constructor, and a notebook process sets it.
_CLIENT_VARS = ("OPERATION_MODE", "ARC_BASE_URL", "ARC_API_KEY", "ENVIRONMENTS_DIR", "RECORDINGS_DIR",
                "ONLY_RESET_LEVELS")


@contextmanager
def _gateway_env(only_reset_levels: bool) -> Iterator[None]:
    saved = {k: os.environ.pop(k) for k in _CLIENT_VARS if k in os.environ}
    if only_reset_levels:
        os.environ["ONLY_RESET_LEVELS"] = "true"
    try:
        yield
    finally:
        for k in _CLIENT_VARS:
            os.environ.pop(k, None)
        os.environ.update(saved)


def prepare_env_dir(src: Path, dst: Path, clone_to: int | None = None) -> list[str]:
    """Copy every game under `src` to `dst`, then add clones up to `clone_to` games. Returns the game ids."""
    metas = sorted(src.rglob("metadata.json"))
    if not metas:
        raise FileNotFoundError(f"no metadata.json under {src}")
    ids: list[str] = []
    originals: list[tuple[Path, dict[str, Any]]] = []
    for m in metas:
        meta = json.loads(m.read_text())
        rel = m.parent.relative_to(src)
        shutil.copytree(m.parent, dst / rel, dirs_exist_ok=True)
        ids.append(meta["game_id"])
        originals.append((m.parent, meta))
    k = 0
    while clone_to is not None and len(ids) < clone_to:
        game_dir, meta = originals[k % len(originals)]
        base = meta["game_id"].split("-", 1)[0]
        version = meta["game_id"].split("-", 1)[1] if "-" in meta["game_id"] else "0"
        new_base = f"k{k:03d}"
        clone = dict(meta)
        clone["game_id"] = f"{new_base}-{version}"
        clone["class_name"] = meta.get("class_name") or base[0].upper() + base[1:4]
        clone.pop("local_dir", None)
        out = dst / new_base / version
        shutil.copytree(game_dir, out, dirs_exist_ok=True)
        (out / "metadata.json").write_text(json.dumps(clone, indent=2))
        ids.append(clone["game_id"])
        k += 1
    return ids


class GatewaySim:
    def __init__(self, env_dir: Path, out_dir: Path, only_reset_levels: bool = False) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("wmagent.gateway_sim")
        logger.setLevel(logging.WARNING)
        with _gateway_env(only_reset_levels):
            self.arcade = Arcade(
                operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir),
                recordings_dir=str(out_dir / "recordings"), logger=logger,
            )
            self.app, self.api = create_app(self.arcade, competition_mode=True, on_scorecard_close=self._on_close)
        if not self.arcade.available_environments:
            raise FileNotFoundError(f"no environments under {env_dir}")
        self.lock = threading.Lock()
        self.status: dict[str, Any] = {
            "games": [e.game_id for e in self.arcade.available_environments],
            "scorecards_opened": 0, "open_rejected": 0, "scorecards_closed": [], "makes": {},
            "closed_scorecard": None,
        }
        self._wrap()
        self.app.add_url_rule("/api/games/anonkey", "anonkey", lambda: jsonify({"api_key": f"anon-{uuid.uuid4()}"}))
        self.app.add_url_rule("/sim/status", "sim_status", self._status_view)
        self._server = None
        self._thread: threading.Thread | None = None

    def _wrap(self) -> None:
        make, new_scorecard = self.arcade.make, self.api.new_scorecard

        def counted_make(game_id: str, *a: Any, **kw: Any) -> Any:
            env = make(game_id, *a, **kw)
            if env is not None:
                with self.lock:
                    gid = env.environment_info.game_id
                    self.status["makes"][gid] = self.status["makes"].get(gid, 0) + 1
            return env

        def counted_open() -> Any:
            resp, code = new_scorecard()
            with self.lock:
                self.status["scorecards_opened" if code == 200 else "open_rejected"] += 1
            return resp, code

        self.arcade.make = counted_make  # type: ignore[method-assign]
        self.api.new_scorecard = counted_open  # type: ignore[method-assign]

    def _on_close(self, scorecard: Any) -> None:
        data = scorecard.model_dump(mode="json", exclude_none=True)
        data.pop("api_key", None)
        summary = {k: data.get(k) for k in ("card_id", "score", "total_environments", "total_environments_completed",
                                             "total_levels", "total_levels_completed", "total_actions")}
        with self.lock:
            self.status["scorecards_closed"].append(data.get("card_id"))
            self.status["closed_scorecard"] = summary
        (self.out_dir / "scorecard_closed.json").write_text(json.dumps(data, indent=2, default=str))

    def _status_view(self) -> Any:
        with self.lock:
            return jsonify(self.status)

    def serve_background(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """Start serving in a daemon thread; returns the base URL (port 0 picks a free port)."""
        self._server = make_server(host, port, self.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{self._server.server_port}"

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True, type=Path, help="environment_files tree to serve")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--clone-to", type=int, default=None)
    ap.add_argument("--only-reset-levels", action="store_true")
    a = ap.parse_args(argv)
    served = a.out_dir / "environment_files"
    ids = prepare_env_dir(a.env_dir, served, a.clone_to)
    sim = GatewaySim(served, a.out_dir, a.only_reset_levels)
    server = make_server(a.host, a.port, sim.app, threaded=True)
    print(f"[gateway_sim] {len(ids)} games on http://{a.host}:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
