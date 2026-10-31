"""No-progress breaker and harness-written level memory (Plan C step 4b; WM_MECHANISMS=all).

Judged from output, never from the model's own report (the health scheduler idea from the earlier harnesses):
the monitor sees every executed action with the board before and after it, every cell's code, and every gate
refusal, and it

- M2a, no-op deny, SHADOW ONLY: on a level, an action whose (board, action) pair already changed nothing is
  logged as `noop_would_deny`, and its real outcome as `noop_outcome`. The 0a back-test on the old logs found
  that 11 of 14 such repeats did change the board (hidden state), so it is never enforced;
- M2b, STUCK line: after STUCK_NO_CHANGE executed actions in a row that changed nothing, STUCK_REPEATS runs
  of an identical cell on the level, or STUCK_UNCAPTURED gate refusals in a row with no predict captured, the
  next user prompt carries one fact line; whether the next 3 actions differ from the repeated ones is logged;
- M4d, level memory: on a completed level it records the action count, the start of the action sequence and
  each label's effect rate; later prompts get a "Solved levels" block, and whether the first 5 actions on the
  next level use labels that had an effect is logged.

Thresholds come from the 0a replay (`scripts/mechanism_baseline.py`): no-change runs had median 1 and p90 2,
identical-cell repeats p90 2, uncaptured refusal streaks at most 2.

Records go to <stem>_progress.jsonl next to the state file.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STUCK_NO_CHANGE = 2
STUCK_REPEATS = 2
STUCK_UNCAPTURED = 2
FOLLOWUP_ACTIONS = 3
MEMORY_SEQUENCE = 20
MEMORY_FIRST_ACTIONS = 5

_KIND = re.compile(r"^([A-Z_0-9]+)")


def board_hash(grid: Any) -> str:
    return hashlib.sha1(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:16]


def action_kind(display: str) -> str:
    """The label without click coordinates: MOUSE(row=3, col=4) -> MOUSE."""
    match = _KIND.match(str(display).strip().upper())
    return match.group(1) if match else str(display)


@dataclass
class LevelStats:
    actions: int = 0
    played: Counter = field(default_factory=Counter)
    changed: Counter = field(default_factory=Counter)
    sequence: list[str] = field(default_factory=list)


@dataclass
class ProgressMonitor:
    log_path: Path | None = None
    level: int | None = None
    # M2a: (board hash, action display) pairs that changed nothing on this level
    no_effect: set[tuple[str, str]] = field(default_factory=set)
    no_change_run: list[str] = field(default_factory=list)
    cell_counts: Counter = field(default_factory=Counter)
    uncaptured_refusals: int = 0
    stuck_line: str | None = None
    followup: dict[str, Any] | None = None
    stats: LevelStats = field(default_factory=LevelStats)
    solved: list[dict[str, Any]] = field(default_factory=list)
    first_actions: list[str] = field(default_factory=list)
    effective_before: set[str] = field(default_factory=set)

    @classmethod
    def for_state(cls, state_path: Path) -> ProgressMonitor:
        stem = state_path.name.removesuffix(".json").removesuffix("_tool_runtime_state")
        return cls(log_path=state_path.with_name(f"{stem}_progress.jsonl"))

    def _log(self, event: str, **fields: Any) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.time(), "event": event, "level": self.level, **fields}) + "\n")

    # --- level bookkeeping -------------------------------------------------

    def _enter_level(self, level: int) -> None:
        if self.level is not None and level != self.level:
            self.no_effect.clear()
            self.no_change_run = []
            self.cell_counts.clear()
            self.uncaptured_refusals = 0
            self.stats = LevelStats()
            self.first_actions = []
        self.level = level

    # --- M2a shadow check, before the actions run --------------------------

    def shadow_check(self, level: int, grid: Any, displays: list[str]) -> list[int]:
        """Positions of the requested actions the no-op deny would refuse: item 1, or a later item when every
        item before it is itself a known no-op on the same board. Logged only."""
        self._enter_level(level)
        current = board_hash(grid)
        hits = []
        for position, display in enumerate(displays):
            if display == "RESET" or (current, display) not in self.no_effect:
                break
            hits.append(position)
        if hits:
            self._log("noop_would_deny", positions=hits, actions=[displays[i] for i in hits], board=current)
        return hits

    # --- outcomes, after the actions ran -------------------------------------

    def record(self, frames: list[dict[str, Any]], displays: list[str], shadow_hits: list[int]) -> None:
        """`frames` are the state-history frames from the board before the first executed action through the
        board after the last; `displays` the executed actions, one per transition."""
        for position, display in enumerate(displays):
            before, after = frames[position], frames[position + 1]
            level = int(before.get("level") or 1)
            self._enter_level(level)
            if display == "RESET":
                self.no_effect.clear()
                continue
            changed = before.get("grid") != after.get("grid")
            if position in shadow_hits:
                self._log("noop_outcome", action=display, changed=changed)
            key = (board_hash(before.get("grid")), display)
            if changed:
                self.no_effect.discard(key)
                self.no_change_run = []
            else:
                self.no_effect.add(key)
                self.no_change_run.append(display)
            kind = action_kind(display)
            self.stats.actions += 1
            self.stats.played[kind] += 1
            self.stats.changed[kind] += changed
            if len(self.stats.sequence) < MEMORY_SEQUENCE:
                self.stats.sequence.append(display)
            self._first_action(kind)
            self._followup_action(display)
            if int(after.get("level") or level) != level:
                self._level_solved(level)
        if len(self.no_change_run) >= STUCK_NO_CHANGE:
            run = Counter(self.no_change_run[-6:])
            self._stuck("no_change", f"Your last {len(self.no_change_run)} actions changed nothing on the board: "
                        + ", ".join(f"{a}x{n}" for a, n in run.most_common(4)) + ".", set(self.no_change_run))

    def record_cell(self, code: str) -> None:
        digest = hashlib.sha1(code.strip().encode()).hexdigest()[:16]
        self.cell_counts[digest] += 1
        repeats = self.cell_counts[digest] - 1
        if repeats >= STUCK_REPEATS:
            self._stuck("repeated_cell", f"You have now run the same python cell {repeats + 1} times on this "
                        "level with the same code; its result will not change.", set())

    def record_refusal(self, captured: bool) -> None:
        self.uncaptured_refusals = 0 if captured else self.uncaptured_refusals + 1
        if self.uncaptured_refusals >= STUCK_UNCAPTURED:
            self._stuck("uncaptured_refusals", f"action() was refused {self.uncaptured_refusals} times in a row "
                        "and no predict was captured in those cells.", set())

    def record_capture(self) -> None:
        self.uncaptured_refusals = 0

    def _stuck(self, signal: str, line: str, repeated: set[str]) -> None:
        if self.stuck_line is None:
            self._log("stuck", signal=signal, line=line)
        self.stuck_line = "Progress check: " + line
        self.followup = {"signal": signal, "repeated": repeated, "next": []}

    def take_stuck_line(self) -> str | None:
        """The STUCK line for the next user prompt, given once."""
        line, self.stuck_line = self.stuck_line, None
        return line

    def _followup_action(self, display: str) -> None:
        if self.followup is None or self.stuck_line is not None:
            return
        self.followup["next"].append(display)
        if len(self.followup["next"]) >= FOLLOWUP_ACTIONS:
            nxt = self.followup["next"]
            self._log("stuck_followup", signal=self.followup["signal"], next_actions=nxt,
                      differed=not (set(nxt) & self.followup["repeated"]))
            self.followup = None

    # --- M4d level memory ---------------------------------------------------

    def _level_solved(self, level: int) -> None:
        rates = {k: round(self.stats.changed[k] / n, 2) for k, n in self.stats.played.items() if n}
        record = {"level": level, "actions": self.stats.actions, "sequence": list(self.stats.sequence),
                  "effect_rates": rates}
        self.solved.append(record)
        self._log("level_memory", **record)
        self.effective_before = {k for k, r in rates.items() if r > 0}

    def _first_action(self, kind: str) -> None:
        if not self.solved or len(self.first_actions) >= MEMORY_FIRST_ACTIONS:
            return
        self.first_actions.append(kind)
        if len(self.first_actions) == MEMORY_FIRST_ACTIONS:
            used = sum(k in self.effective_before for k in self.first_actions)
            self._log("memory_use", first_actions=self.first_actions, effective_labels=sorted(self.effective_before),
                      used_effective=used)

    def memory_block(self) -> str | None:
        """The "Solved levels" block for later prompts. A label with no effect is marked by how many solved
        levels showed that (a single level is weak evidence)."""
        if not self.solved:
            return None
        lines = ["Solved levels (recorded by the harness):"]
        no_effect: Counter = Counter()
        for rec in self.solved:
            shown = ", ".join(rec["sequence"][:10]) + (" ..." if rec["actions"] > 10 else "")
            effects = ", ".join(f"{k} {int(r * 100)}%" for k, r in sorted(rec["effect_rates"].items()))
            lines.append(f"- Level {rec['level']}: {rec['actions']} actions ({shown}). Board changed by: {effects}.")
            no_effect.update(k for k, r in rec["effect_rates"].items() if r == 0)
        if no_effect:
            marks = ", ".join(f"{k} ({n} level{'s' if n > 1 else ''})" for k, n in sorted(no_effect.items()))
            lines.append(f"No effect observed: {marks}.")
        return "\n".join(lines)
