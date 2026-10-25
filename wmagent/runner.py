"""Runner: starts every game copy, plays them on a fixed thread pool, writes the results (spec 2.2).

The runner does not know how a game is played. It hands each started game to a
`play(ctx)` callable (the session, spec 2.3) together with a `PlayContext` that
carries the artifact paths, the stop flag, the per-game time cap and the
request-timeout rule. Everything about ordering, admission, deadlines, error
isolation, teardown and `benchmark.json` lives here.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .engine import COMPETITION, OFFLINE, PLAYING, WON, Engine, Game

HEARTBEAT_S = 600.0
MIN_REQUEST_TIMEOUT_S = 0.1


@dataclass
class RunConfig:
    label: str
    game_ids: list[str] | str
    n_passes: int = 1
    concurrency: int = 10
    max_runtime_s_per_game: float | None = 2700.0
    max_actions_per_game: int | None = None
    analyzer_timeout_s: float = 900.0
    cancel_drain_timeout_s: float = 120.0
    max_runtime_s: float = 32400.0
    soft_deadline_buffer_s: float = 600.0
    environments_dir: str | None = None
    solver_label: str = "wmagent"
    # Submission mode (spec 3.4 R3, a deliberate deviation): close everything before the notebook limit.
    submission_margin_s: float = 900.0
    derive_per_game_cap: bool = True

    @property
    def competition(self) -> bool:
        return self.game_ids == COMPETITION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown run config keys: {sorted(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if not self.competition:
            if not isinstance(self.game_ids, list) or not self.game_ids:
                raise ValueError("game_ids must be a non-empty list or 'competition'")
            if len(set(self.game_ids)) != len(self.game_ids):
                raise ValueError(f"duplicate game ids within one pass: {self.game_ids}")
        if self.n_passes < 1 or self.concurrency < 1:
            raise ValueError("n_passes and concurrency must be >= 1")


@dataclass
class PlayContext:
    """What the session needs to play one game copy."""

    game: Game
    stem: str
    state_path: Path
    transcript_path: Path
    events_path: Path
    stop_event: threading.Event
    per_game_cap_s: float | None
    soft_deadline: float | None  # time.monotonic() value
    max_actions: int | None
    analyzer_timeout_s: float
    play_started: float = field(default_factory=time.monotonic)

    def time_remaining(self) -> float | None:
        if self.per_game_cap_s is None:
            return None
        return self.per_game_cap_s - (time.monotonic() - self.play_started)

    def time_cap_reached(self) -> bool:
        remaining = self.time_remaining()
        return remaining is not None and remaining <= 0

    def action_cap_reached(self) -> bool:
        return self.max_actions is not None and len(self.game.history) >= self.max_actions

    def should_stop(self) -> bool:
        return (
            self.game.state != PLAYING
            or self.stop_event.is_set()
            or self.time_cap_reached()
            or self.action_cap_reached()
        )

    def request_timeout(self) -> float:
        limits = [self.analyzer_timeout_s]
        remaining = self.time_remaining()
        if remaining is not None:
            limits.append(remaining)
        if self.soft_deadline is not None:
            limits.append(self.soft_deadline - time.monotonic())
        return max(MIN_REQUEST_TIMEOUT_S, min(limits))


PlayFn = Callable[[PlayContext], None]


@dataclass
class RunResult:
    games: list[Game]
    start_time: datetime
    end_time: datetime
    benchmark_path: Path | None


def plan_runs(game_ids: list[str], n_passes: int) -> list[tuple[str, str]]:
    """Passes-major order: every pass-0 copy, then pass 1, ...; stems are <game_id>_p<k>."""
    seen: dict[str, int] = {}
    runs = []
    for _ in range(n_passes):
        for game_id in game_ids:
            k = seen.get(game_id, 0)
            seen[game_id] = k + 1
            runs.append((game_id, f"{game_id}_p{k}"))
    return runs


def soft_deadline_epoch(config: RunConfig, notebook_start_epoch: float, submission: bool) -> float:
    if submission:
        return notebook_start_epoch + config.max_runtime_s - config.submission_margin_s
    return notebook_start_epoch + config.max_runtime_s - min(config.soft_deadline_buffer_s, config.max_runtime_s / 2)


def summary_text(label: str, games: list[Game], n_passes: int, start: datetime, end: datetime | None) -> str:
    end = end or datetime.now()
    scores = [g.final_score for g in games if g.final_score is not None]
    lines = [
        f"=== {label} ===",
        (
            f"games: {len({g.game_id for g in games})}  passes: {n_passes}  runs: {len(games)}  "
            f"wins: {sum(g.state == WON for g in games)}"
        ),
        f"start: {start.isoformat()}  end: {end.isoformat()}  duration: {(end - start).total_seconds():.0f}s",
        (
            f"mean score: {statistics.fmean(scores) if scores else 0.0:.3f}  "
            f"median score: {statistics.median(scores) if scores else 0.0:.3f}"
        ),
        f"total actions: {sum(len(g.history) for g in games)}  total tokens: {sum(g.generated_tokens for g in games)}",
    ]
    return "\n".join(lines)


def benchmark_json(config: RunConfig, games: list[Game], n_passes: int, start: datetime, end: datetime) -> dict:
    return {
        "label": config.label,
        "n_passes": n_passes,
        "solver_label": config.solver_label,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "game_runs": [g.to_json() for g in games],
    }


class Runner:
    def __init__(
        self,
        config: RunConfig,
        play: PlayFn,
        output_dir: str | Path,
        *,
        engine: Engine | None = None,
        submission: bool = False,
        notebook_start_epoch: float | None = None,
        heartbeat_s: float = HEARTBEAT_S,
    ) -> None:
        config.validate()
        self.config = config
        self.play = play
        self.output_dir = Path(output_dir)
        self.submission = submission
        self.stop_event = threading.Event()
        if engine is None:
            if config.competition:
                engine = Engine(COMPETITION, stop_event=self.stop_event)
            else:
                engine = Engine(OFFLINE, environments_dir=config.environments_dir, stop_event=self.stop_event)
        else:
            engine.stop_event = self.stop_event
        self.engine = engine
        if notebook_start_epoch is None:
            notebook_start_epoch = float(os.getenv("NOTEBOOK_START_EPOCH") or time.time())
        self.notebook_start_epoch = notebook_start_epoch
        self.heartbeat_s = heartbeat_s
        self.games: list[Game] = []
        self.contexts: dict[int, PlayContext] = {}
        self.start_time = datetime.now()

    # ------------------------------------------------------------- timing
    def _soft_deadline_mono(self) -> float:
        epoch = soft_deadline_epoch(self.config, self.notebook_start_epoch, self.submission)
        return time.monotonic() + (epoch - time.time())

    def _per_game_cap(self, deadline_mono: float, n_runs: int) -> float | None:
        cap = self.config.max_runtime_s_per_game
        if self.submission and self.config.derive_per_game_cap:
            waves = math.ceil(n_runs / self.config.concurrency)
            budget = max(0.0, deadline_mono - time.monotonic()) / max(1, waves)
            cap = budget if cap is None else min(cap, budget)
        return cap

    # ---------------------------------------------------------------- run
    def run(self) -> RunResult:
        self.start_time = datetime.now()
        if self.config.competition:
            game_ids = self.engine.available_game_ids()
            if not game_ids:
                raise RuntimeError("competition mode: no available environments")
            n_passes = 1
        else:
            game_ids = list(self.config.game_ids)
            n_passes = self.config.n_passes
        runs = plan_runs(game_ids, n_passes)
        for sub in ("artifacts", "transcripts"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        deadline = self._soft_deadline_mono()
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(heartbeat_stop, n_passes), daemon=True)
        heartbeat.start()
        try:
            self._start_all(runs)
            cap = self._per_game_cap(deadline, len(runs))
            self._play_all(deadline, cap)
        except Exception as exc:  # noqa: BLE001 - a solver-level error is printed, teardown still runs
            print(f"[runner] solver error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            heartbeat_stop.set()
            result = self._teardown(n_passes)
        return result

    def _start_all(self, runs: list[tuple[str, str]]) -> None:
        """Make every copy serially before any play, as the competition scorecard expects."""
        for game_id, stem in runs:
            game = self.engine.new_game(game_id)
            game.label = stem
            self.games.append(game)
            try:
                game.start()
            except Exception as exc:  # noqa: BLE001 - error isolation: only this copy crashes
                game.mark_crashed(f"error: {type(exc).__name__}: {exc}")
        for game in self.games:
            if game.state != PLAYING:
                game.finish()  # after all starts: releasing earlier could close the shared scorecard

    def _context(self, game: Game, deadline: float, cap: float | None) -> PlayContext:
        artifacts = self.output_dir / "artifacts"
        return PlayContext(
            game=game,
            stem=game.label,
            state_path=artifacts / f"{game.label}_tool_runtime_state.json",
            transcript_path=self.output_dir / "transcripts" / f"{game.label}.txt",
            events_path=artifacts / f"{game.label}_events.jsonl",
            stop_event=self.stop_event,
            per_game_cap_s=cap,
            soft_deadline=deadline,
            max_actions=self.config.max_actions_per_game,
            analyzer_timeout_s=self.config.analyzer_timeout_s,
        )

    def _play_one(self, game: Game, deadline: float, cap: float | None) -> None:
        if game.finished:
            return
        ctx = self._context(game, deadline, cap)
        self.contexts[id(game)] = ctx
        try:
            if not ctx.should_stop():
                self.play(ctx)
        except Exception as exc:  # noqa: BLE001 - error isolation: only this game crashes
            game.mark_crashed(f"error: {type(exc).__name__}: {exc}")
        finally:
            game.finish()

    def _play_all(self, deadline: float, cap: float | None) -> None:
        pool = ThreadPoolExecutor(max_workers=self.config.concurrency, thread_name_prefix="game")
        try:
            futures = {pool.submit(self._play_one, g, deadline, cap): g for g in self.games}
            _, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
            if pending:
                print(f"[runner] soft deadline reached; stopping {len(pending)} unfinished game(s)", flush=True)
                self.stop_event.set()
                _, pending = wait(pending, timeout=self.config.cancel_drain_timeout_s)
                for future in pending:
                    future.cancel()
                    futures[future].finish()  # stop flag is set, so it finishes as cancelled
            for future, game in futures.items():  # a failed scorecard close surfaces here; _teardown retries it
                if future.done() and not future.cancelled() and future.exception() is not None:
                    exc = future.exception()
                    print(f"[runner] {game.game_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _teardown(self, n_passes: int) -> RunResult:
        for game in self.games:
            if game.state == PLAYING and not game.finished:
                game.mark_crashed("error: RuntimeError: still playing at teardown")
            game.finish()
        if self.engine.scorecard is not None:
            self.engine.scorecard.close()
        end = datetime.now()
        path = None
        if not self.submission:
            path = self.output_dir / "benchmark.json"
            data = benchmark_json(self.config, self.games, n_passes, self.start_time, end)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)
        print(summary_text(self.config.label, self.games, n_passes, self.start_time, end), flush=True)
        return RunResult(list(self.games), self.start_time, end, path)

    def _heartbeat(self, stop: threading.Event, n_passes: int) -> None:
        while not stop.wait(self.heartbeat_s):
            print(summary_text(self.config.label, self.games, n_passes, self.start_time, None), flush=True)


def build_runner(
    config: RunConfig,
    output_dir: str | Path,
    agent_factory: Callable[[], Any],
    gate_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> Runner:
    """Wire the runner to the session (spec 2.3): play = session.make_play(agent_factory, gate_factory)."""
    from .session import make_play

    return Runner(config, make_play(agent_factory, gate_factory), output_dir, **kwargs)
