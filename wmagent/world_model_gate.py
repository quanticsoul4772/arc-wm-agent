"""Enforced executable world model for the B feasibility probe.

When PROBE_ENFORCE=1, the harness:
- captures top-level definitions from any successful python cell that defines
  `predict(ascii, action)` (anchored match, executed code only), and keeps them
  in a per-game store that is prepended to every later cell;
- backtests the stored `predict` on recorded transitions, using only the
  informative ones (those where more than CHANGE_FLOOR cells changed, so an
  identity model cannot pass) and scoring a hit when the
  prediction is within MISS_TOLERANCE cells and beats predicting no change;
- scores the accepted model prequentially on transitions observed after it was
  accepted (held-out by time);
- in shadow mode, judges each acceptance on its first HELDOUT_N informative
  transitions after acceptance, which the model could not have seen: it is
  confirmed only if it hits HELDOUT_PASS_HITS of them (3 of 3; ">= 75% of 3"
  in the plan means the same), and is a false confidence if the first is a
  miss; the verdict is logged but never gates actions;
- after PROBE_ACTIONS real actions on a level, refuses `action()` until a model
  passes, or until MAX_REFUSALS refusals release the level as sim-free; a batch
  that would cross PROBE_ACTIONS without a model is capped;
- seeds an identity skeleton `predict` (which can never pass) for the model to
  replace, and judges models per level, skipping RESET transitions and the
  transition just before a RESET (game over or a chosen reset);
- with PROBE_OPTIONAL=1 (method C), never refuses actions: writing `predict` is the
  model's choice, and a `simulate(actions)` helper that previews boards with
  `predict` works only after the model passed the held-out check;
- with PROBE_RELEASE_AFTER_ACCEPT=1, stops refusing actions for the rest of the
  run once any model has been accepted (prequential and held-out scoring keep
  logging in shadow), so the gate's cost is paid at most once per run.

Every decision is appended to <run_stem>_wm_gate.jsonl next to the run's state file.
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wmagent.game_class import GateProfile, classify, profile_for, shadow_profiles

PREDICT_PATTERN = re.compile(r"^def predict\(", re.MULTILINE)

CHANGE_FLOOR = 2
MISS_TOLERANCE = 2
PASS_ACCURACY = 0.75
MIN_INFORMATIVE = 3
PREQUENTIAL_WINDOW = 10
PREQUENTIAL_FLOOR = 0.5
HELDOUT_N = 3
HELDOUT_PASS_HITS = 3

# WM_MECHANISMS (Plan C arms): "base" is the harness as it was, "m1a" adds capture-on-deny and the
# stated acceptance rule, "all" adds every Plan C mechanism, "all-nocont" is "all" without the yield-continuation
# prompt (M2c; arm C slowed down on it, C201). Read only in from_env / AgentConfig.from_env.
MECHANISMS = ("base", "m1a", "all", "all-nocont")
FULL_MECHANISMS = ("all", "all-nocont")


def mechanisms_from_env() -> str:
    value = os.environ.get("WM_MECHANISMS", "").strip().lower() or "base"
    if value not in MECHANISMS:
        raise ValueError(f"WM_MECHANISMS must be one of {MECHANISMS}, got {value!r}")
    return value

SKELETON_PREDICT = '''def predict(ascii, action):
    # TODO: replace with the game's dynamics: return the board after `action`.
    return ascii'''

# Method C: prepended to every cell. Previews boards with the stored `predict`
# without spending real actions, but only once the model passed the held-out check.
SIMULATE_HELPER = '''def simulate(actions, ascii=None):
    """Predicted boards after each action in `actions`, from `ascii` (default: the
    current board), using your `predict`. Spends no real actions. Works only after
    `predict` has been confirmed on the 3 informative moves made after it was accepted."""
    if not __WM_CONFIRMED:
        return "simulate unavailable: predict is not confirmed yet (it must match the next 3 informative real moves after acceptance)"
    board = current_frame.ascii if ascii is None else ascii
    boards = []
    for a in actions:
        board = predict(board, a)
        boards.append(board)
    return boards'''


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def extract_definitions(code: str) -> dict[str, str] | None:
    """Top-level imports, defs, classes and constant assignments from a cell
    that defines predict; None if the cell does not define predict."""
    if not PREDICT_PATTERN.search(code):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    definitions: dict[str, str] = {}
    for node in tree.body:
        segment = ast.get_source_segment(code, node)
        if segment is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = segment
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            definitions[f"import:{segment}"] = segment
        elif (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
              and not any(isinstance(n, ast.Call) for n in ast.walk(node.value))):
            for target in node.targets:
                definitions[f"const:{target.id}"] = segment
    return definitions if "predict" in definitions else None


def cell_mismatch(predicted: str, actual: str) -> int:
    """Count of differing cells between two ascii boards (whitespace-trimmed rows)."""
    p_rows = [row.strip() for row in str(predicted).strip().splitlines()]
    a_rows = [row.strip() for row in str(actual).strip().splitlines()]
    mismatches = abs(len(p_rows) - len(a_rows)) * max((len(r) for r in a_rows), default=0)
    for p_row, a_row in zip(p_rows, a_rows):
        mismatches += abs(len(p_row) - len(a_row))
        mismatches += sum(1 for p, a in zip(p_row, a_row) if p != a)
    return mismatches


def _is_hit(changed: int, miss: int | None) -> bool:
    """A prediction counts if it is within a small tolerance (timer bars, a
    stray cell) and strictly better than predicting no change."""
    if miss is None:
        return False
    return miss <= max(MISS_TOLERANCE, 0.05 * changed) and miss < changed


# Runs inside the restricted sandbox after the stored definitions.
BACKTEST_DRIVER = r'''
def __probe_mismatch(predicted, actual):
    p_rows = [row.strip() for row in str(predicted).strip().splitlines()]
    a_rows = [row.strip() for row in str(actual).strip().splitlines()]
    count = abs(len(p_rows) - len(a_rows)) * max([len(r) for r in a_rows] or [0])
    for p_row, a_row in zip(p_rows, a_rows):
        count += abs(len(p_row) - len(a_row))
        count += sum(1 for p, a in zip(p_row, a_row) if p != a)
    return count

__probe_rows = []
for __i, __t in enumerate(transitions):
    if __t.before_frame is None:
        continue
    if __t.before_frame.level != __t.after_frame.level:
        continue
    if str(__t.action).strip().upper().startswith('RESET'):
        continue
    if __i + 1 < len(transitions) and str(transitions[__i + 1].action).strip().upper().startswith('RESET'):
        continue
    __changed = __probe_mismatch(__t.before_frame.ascii, __t.after_frame.ascii)
    try:
        __pred = predict(__t.before_frame.ascii, __t.action)
        __miss = __probe_mismatch(__pred, __t.after_frame.ascii)
        __err = None
    except Exception as __exc:
        __miss = None
        __err = type(__exc).__name__
    __probe_rows.append([__i, __changed, __miss, __err, __t.after_frame.level, str(__t.action)[:40],
                         __t.before_frame.level])
result = __probe_rows
'''


@dataclass
class WorldModelGate:
    probe_actions: int
    max_refusals: int
    log_path: Path | None = None
    store: dict[str, str] = field(default_factory=dict)
    accepted_at: int | None = None
    accepted_level: int | None = None
    level_actions: int = 0
    level: int | None = None
    refusals: int = 0
    released: bool = False
    prequential: list[bool] = field(default_factory=list)
    scored_through: int = -1
    heldout: list[bool] = field(default_factory=list)
    release_after_accept: bool = False
    ever_accepted: bool = False
    optional: bool = False
    confirmed: bool = False
    mechanisms: str = "base"
    game_class: str | None = None
    # Outcome of the last after_cell: None when no predict was captured, else the verdict dict
    # (with "passed"); read by the agent to tell the model what happened to its predict.
    last_verdict: dict[str, Any] | None = None
    # WM_MECHANISMS=all (Plan C step 4a): the level's last judged verdict (M1b evidence), the carried
    # probation grant (M1c) and the model calls spent on a blocked level (M1d).
    level_verdict: dict[str, Any] | None = None
    grant: bool = False
    grant_misses: int = 0
    grant_scored_through: int = -1
    blocked_calls: int = 0
    deny_followup: dict[str, Any] | None = None
    submitted_predict: str | None = None

    @classmethod
    def from_env(cls) -> WorldModelGate | None:
        if os.environ.get("PROBE_ENFORCE", "").strip() != "1":
            return None
        return cls(
            probe_actions=_env_int("PROBE_ACTIONS", 30),
            max_refusals=_env_int("PROBE_MAX_REFUSALS", 8),
            release_after_accept=os.environ.get("PROBE_RELEASE_AFTER_ACCEPT", "").strip() == "1",
            optional=os.environ.get("PROBE_OPTIONAL", "").strip() == "1",
            mechanisms=mechanisms_from_env(),
        )

    @property
    def evaluate_on_deny(self) -> bool:
        return self.mechanisms != "base"

    @property
    def full(self) -> bool:
        """WM_MECHANISMS=all / all-nocont: deny evidence (M1b), carried grant (M1c), release valve (M1d)."""
        return self.mechanisms in FULL_MECHANISMS

    def reset(self, state_path: Path) -> None:
        stem = state_path.name.removesuffix(".json").removesuffix("_tool_runtime_state")
        self.log_path = state_path.with_name(f"{stem}_wm_gate.jsonl")
        self.store = {"predict": SKELETON_PREDICT}
        self.accepted_at = None
        self.accepted_level = None
        self.level_actions = 0
        self.level = None
        self.refusals = 0
        self.released = False
        self.prequential = []
        self.scored_through = -1
        self.heldout = []
        self.ever_accepted = False
        self.confirmed = False
        self.last_verdict = None
        self.level_verdict = None
        self.grant = False
        self.grant_misses = 0
        self.grant_scored_through = -1
        self.blocked_calls = 0
        self.deny_followup = None
        self.submitted_predict = None
        # The manifest: what this run actually enforces, so each arm is confirmed from its logs.
        self._log(
            "session_start",
            probe_actions=self.probe_actions,
            max_refusals=self.max_refusals,
            release_after_accept=self.release_after_accept,
            optional=self.optional,
            mechanisms=self.mechanisms,
            change_floor=self.base_profile.change_floor,
            min_informative=self.base_profile.min_informative,
            pass_accuracy=self.base_profile.pass_accuracy,
        )

    def _log(self, event: str, **fields: Any) -> None:
        if self.log_path is None:
            return
        record = {"t": time.time(), "event": event, "game_class": self.game_class, **fields}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def verdict_note(self) -> str | None:
        """What happened to the predict captured by the last cell, for a cell whose action() was refused:
        the refusal text was written before the backtest ran, so it says nothing about this predict."""
        verdict = self.last_verdict
        if verdict is None:
            return None
        if verdict["passed"]:
            return "World-model gate: your predict was accepted on this level; action() is unblocked."
        if verdict.get("reason") == "backtest_failed":
            return "World-model gate: your predict was captured but its backtest failed to run."
        return (
            f"World-model gate: your predict was captured but not accepted: it matched "
            f"{round(verdict['accuracy'] * verdict['informative'])} of {verdict['informative']} informative moves "
            f"(needs at least {self.profile.min_informative} moves that change more than "
            f"{self.profile.change_floor} cells, and at least {int(self.profile.pass_accuracy * 100)}% of them matched)."
        )

    def note_backtest_error(self, error: str) -> None:
        self._log("backtest_error", level=self.level, error=str(error)[:300])

    def stored_source(self) -> str:
        return "\n\n".join(self.store.values())

    def prepend(self, code: str) -> str:
        source = self.stored_source()
        if self.optional:
            source = f"{source}\n\n__WM_CONFIRMED = {self.confirmed!r}\n{SIMULATE_HELPER}"
        return f"{source}\n\n{code}" if source else code

    # --- level and action accounting -------------------------------------

    def observe_level(self, level: int | None) -> None:
        if level is None or level == self.level:
            return
        if self.level is not None:
            self._log("level_change", from_level=self.level, to_level=level)
        # M1c: a model confirmed on held-out moves of the level just finished carries a probation grant
        # into this one level; it is renewed only by a new confirmation here.
        self.grant = self.full and self.level is not None and self.confirmed
        self.grant_misses = 0
        if self.grant:
            self._log("grant", from_level=self.level, to_level=level)
        self.level_verdict = None
        self.blocked_calls = 0
        self.deny_followup = None
        self.level = level
        self.level_actions = 0
        self.refusals = 0
        self.released = False
        self.accepted_at = None
        self.accepted_level = None
        self.prequential = []
        self.heldout = []
        self.confirmed = False

    # --- profile (M3: one source for the gate, the prompt and the logs) ---

    @property
    def base_profile(self) -> GateProfile:
        return GateProfile(probe_actions=self.probe_actions, change_floor=CHANGE_FLOOR,
                           min_informative=MIN_INFORMATIVE, pass_accuracy=PASS_ACCURACY,
                           blocked_call_release=self.max_refusals, forced=not self.optional)

    @property
    def profile(self) -> GateProfile:
        return profile_for(self.game_class, self.base_profile)

    def observe_class(self, valid_actions: list[str]) -> None:
        # No event of its own (the event sequence stays as before): every record carries `game_class`,
        # so a change shows in the field of the next record.
        game_class = classify(valid_actions)
        if game_class is not None:
            self.game_class = game_class

    def _shadow(self, requested: int) -> dict[str, bool]:
        """Would each shadow profile also refuse this request? Logged only, never enforced."""
        out = {}
        for name, alt in shadow_profiles(self.profile).items():
            remaining = alt.probe_actions - self.level_actions
            out[name] = alt.forced and requested > max(0, remaining)
        return out

    def check_action(self, requested: int) -> str | None:
        """None if the actions may run; otherwise the refusal message."""
        profile = self.profile
        if not profile.forced:
            return None
        if self.released or self._model_ok() or self.grant:
            return None
        if self.release_after_accept and self.ever_accepted:
            return None
        remaining = profile.probe_actions - self.level_actions
        if remaining > 0:
            if requested <= remaining:
                return None
            self._log("batch_capped", level=self.level, requested=requested, remaining=remaining,
                      shadow=self._shadow(requested))
            return (
                f"World-model gate: at most {remaining} more action(s) are allowed on this level "
                "before an accepted `predict(ascii, action)` is required. Send a smaller batch."
            )
        self.refusals += 1
        if self.refusals > self.max_refusals:
            self.released = True
            self._log("released_sim_free", level=self.level, refusals=self.refusals - 1)
            return None
        self._log("refused", level=self.level, refusal=self.refusals, requested=requested,
                  shadow=self._shadow(requested))
        if self.full:
            # compared with the predict the model last submitted, not the stored (accepted) one
            self.deny_followup = {"predict": self.submitted_predict,
                                  "accuracy": (self.level_verdict or {}).get("accuracy")}
        return (
            "World-model gate: action() is blocked. Define `def predict(ascii, action)` in a python "
            "cell (top level) that returns the next board ascii for a given board ascii and action "
            "string; the harness backtests it on recorded transitions and unblocks actions once it "
            f"predicts at least {int(profile.pass_accuracy * 100)}% of informative transitions. "
            f"Refusal {self.refusals} of {self.max_refusals}."
        ) + self._deny_evidence()

    def evidence_line(self) -> str:
        """M1b: the level's last verdict in one short line (250 characters at most)."""
        p = self.profile
        rule = f"needs {p.min_informative}+ moves changing >{p.change_floor} cells, {int(p.pass_accuracy * 100)}% matched"
        v = self.level_verdict
        if v is None:
            return f"No predict has been judged on this level yet ({rule})."
        if v.get("reason") == "backtest_failed":
            return f"Your last predict's backtest failed to run ({rule})."
        line = (f"Your last predict matched {round(v['accuracy'] * v['informative'])} of {v['informative']} "
                f"informative moves ({rule}).")
        miss = v.get("first_miss")
        if miss:
            off = "raised an error" if miss["miss"] is None else f"was off by {miss['miss']} cells"
            line += f" First miss: {miss['action']} changed {miss['changed']} cells and predict {off}."
        return line[:250]

    def _deny_evidence(self) -> str:
        return (" " + self.evidence_line()) if self.full else ""

    def blocked(self) -> bool:
        """The level is blocked: it has refused an action and would still refuse the next one."""
        if not self.profile.forced or self.refusals == 0 or self.released or self.grant or self._model_ok():
            return False
        return not (self.release_after_accept and self.ever_accepted)

    def blocked_note(self, level: int | None) -> str | None:
        """M1b re-injection: the evidence line for the next user prompt while the level is blocked. `level` is
        the board's level now; the gate sees a new level only at the next action() call, so a stale level
        means the block it remembers is over."""
        if not self.full or level != self.level or not self.blocked():
            return None
        return "World-model gate: action() is blocked on this level. " + self.evidence_line()

    def note_model_call(self) -> None:
        """M1d release valve: after `blocked_call_release` model calls on a blocked level with no accepted
        model, the level is released. It applies alongside the refusal cap; whichever comes first releases."""
        if not self.full or not self.blocked():
            return
        self.blocked_calls += 1
        if self.blocked_calls >= self.profile.blocked_call_release:
            self.released = True
            self._log("released_blocked_calls", level=self.level, calls=self.blocked_calls,
                      refusals=self.refusals)

    def _open_for_probes(self) -> bool:
        if not self.profile.forced or self.released or self.grant or self._model_ok():
            return False
        return not (self.release_after_accept and self.ever_accepted)

    def batch_allowance(self, requested: int) -> int:
        """M4c (WM_MECHANISMS=all): a batch longer than the probe actions left runs truncated to them instead
        of being refused whole, which would cost a model call; other arms keep the old cap."""
        if not self.full or not self._open_for_probes():
            return requested
        remaining = self.profile.probe_actions - self.level_actions
        if 0 < remaining < requested:
            self._log("batch_truncated", level=self.level, requested=requested, allowed=remaining)
            return remaining
        return requested

    def probe_line(self, level: int | None) -> str | None:
        """M4c: the probe actions left on the board's level, for the user prompt (WM_MECHANISMS=all)."""
        if not self.full:
            return None
        if self.level is not None and level != self.level:
            if self.confirmed:
                return None  # the next action() hands this level a grant: no probe limit applies
            remaining = self.profile.probe_actions  # a new level the gate has not seen yet
        elif not self._open_for_probes():
            return None
        else:
            remaining = self.profile.probe_actions - self.level_actions
        if remaining <= 0:
            return None
        return (f"World-model gate: {remaining} probe action(s) left on this level before your predict must be "
                "accepted. Send them together in one action([...]) list of moves that change the board, then "
                "check the result.")

    def record_actions(self, executed: int) -> None:
        self.level_actions += max(0, executed)

    def _model_ok(self) -> bool:
        if self.accepted_at is None:
            return False
        recent = self.prequential[-PREQUENTIAL_WINDOW:]
        if len(recent) < self.profile.min_informative:
            return True
        return sum(recent) / len(recent) >= PREQUENTIAL_FLOOR

    # --- capture, backtest, prequential scoring --------------------------

    def after_cell(
        self,
        code: str,
        errored: bool,
        run_backtest: Callable[[str], list[list[Any]] | None],
    ) -> None:
        self.last_verdict = None
        if self.grant:
            # M1c: the carried model is scored after every cell while the grant holds, whatever the cell did
            # (a cell that redefines predict would otherwise skip the revoke check for the whole level).
            self._track_grant(run_backtest(self.stored_source()))
        captured = None if errored else extract_definitions(code)
        if captured is not None:
            candidate = dict(self.store)
            candidate.update(captured)
            rows = run_backtest("\n\n".join(candidate.values()))
            verdict = self._judge(rows)
            self.last_verdict = verdict
            self.level_verdict = verdict
            self._log("captured", level=self.level, names=sorted(captured), **verdict)
            if self.deny_followup is not None:
                before = self.deny_followup
                self._log("deny_followup", level=self.level,
                          predict_changed=captured.get("predict") != before["predict"],
                          accuracy_before=before["accuracy"], accuracy_after=verdict.get("accuracy"))
                self.deny_followup = None
            self.submitted_predict = captured.get("predict")
            if verdict["passed"]:
                self.store = candidate
                self.accepted_at = verdict["n_transitions"]
                self.accepted_level = self.level
                self.prequential = []
                self.heldout = []
                self.confirmed = False
                self.scored_through = verdict["n_transitions"] - 1
                self.ever_accepted = True
                self._end_grant_on_accept()
                self._log("accepted", level=self.level, at_transition=self.accepted_at)
            return
        if self.accepted_at is not None:
            rows = run_backtest(self.stored_source())
            self._score_prequential(rows)
            return
        if self.store.get("predict") != SKELETON_PREDICT:
            # A model carried over from an earlier level: re-judge it on this level.
            rows = run_backtest(self.stored_source())
            verdict = self._judge(rows)
            if verdict["passed"]:
                self.accepted_at = verdict["n_transitions"]
                self.accepted_level = self.level
                self.prequential = []
                self.heldout = []
                self.confirmed = False
                self.scored_through = verdict["n_transitions"] - 1
                self.ever_accepted = True
                self._end_grant_on_accept()
                self._log("accepted", level=self.level, at_transition=self.accepted_at, carried=True)

    def _end_grant_on_accept(self) -> None:
        """An accepted model supersedes the grant (prequential scoring takes over), and the release valve
        counts only calls made after the new acceptance."""
        if self.grant:
            self.grant = False
            self._log("grant_superseded", level=self.level)
        self.blocked_calls = 0

    def _track_grant(self, rows: list[list[Any]] | None) -> None:
        """M1c: two consecutive informative misses of the carried model on this level revoke the grant; only
        the flag is cleared, so the normal gate resumes from the level's action count. Only moves played
        within this level count: not the move that completed the last level, nor a RESET's board."""
        for row in rows or []:
            within = len(row) > 6 and row[4] == self.level and row[6] == self.level
            if row[0] <= self.grant_scored_through or not within:
                continue
            self.grant_scored_through = row[0]
            if row[1] <= self.profile.change_floor:
                continue
            self.grant_misses = 0 if _is_hit(row[1], row[2]) else self.grant_misses + 1
            if self.grant_misses >= 2:
                self.grant = False
                self._log("grant_revoked", level=self.level, transition=row[0])
                return

    def _judge(self, rows: list[list[Any]] | None) -> dict[str, Any]:
        if rows is None:
            return {"passed": False, "reason": "backtest_failed", "n_transitions": 0}
        scoped = [r for r in rows if self.level is None or len(r) < 5 or r[4] == self.level]
        profile = self.profile
        informative = [r for r in scoped if r[1] > profile.change_floor]
        hits = [r for r in informative if _is_hit(r[1], r[2])]
        misses = [r for r in informative if not _is_hit(r[1], r[2])]
        first_miss = ({"transition": misses[0][0], "action": misses[0][5] if len(misses[0]) > 5 else "?",
                       "changed": misses[0][1], "miss": misses[0][2]} if misses else None)
        accuracy = len(hits) / len(informative) if informative else 0.0
        n_transitions = (max(r[0] for r in rows) + 1) if rows else 0
        return {
            "passed": len(informative) >= profile.min_informative and accuracy >= profile.pass_accuracy,
            "accuracy": round(accuracy, 3),
            "informative": len(informative),
            "errors": sum(1 for r in rows if r[3]),
            "n_transitions": n_transitions,
            "first_miss": first_miss,
        }

    def _score_prequential(self, rows: list[list[Any]] | None) -> None:
        if not rows:
            return
        for row in rows:
            index, changed, miss, err = row[:4]
            if index <= self.scored_through:
                continue
            self.scored_through = index
            if changed <= self.profile.change_floor:
                continue
            hit = _is_hit(changed, miss)
            self.prequential.append(hit)
            self._log("prequential", level=self.level, transition=index, hit=hit, changed=changed, miss=miss, error=err)
            self._record_heldout(hit)

    def _record_heldout(self, hit: bool) -> None:
        if len(self.heldout) >= HELDOUT_N:
            return
        self.heldout.append(hit)
        if len(self.heldout) < HELDOUT_N:
            return
        hits = sum(self.heldout)
        self.confirmed = hits >= HELDOUT_PASS_HITS
        self._log(
            "heldout",
            level=self.level,
            accepted_at=self.accepted_at,
            heldout_hits=hits,
            heldout_n=HELDOUT_N,
            confirmed=hits >= HELDOUT_PASS_HITS,
            false_confidence=not self.heldout[0],
        )
