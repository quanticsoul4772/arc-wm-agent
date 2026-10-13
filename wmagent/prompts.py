"""Model-facing text for the agent (spec 2.4, T4 topic list).

Written fresh from the topic list; only functional identifiers (runtime globals,
action labels, note labels, `predict`/`simulate`) are shared with other harnesses.
"""
from __future__ import annotations

from typing import Any

# ASCII character -> palette index and colour name (index order = palette order).
COLOR_LEGEND: list[tuple[int, str, str]] = [
    (0, "W", "white"),
    (1, "w", "light grey"),
    (2, "g", "mid grey"),
    (3, "G", "dark grey"),
    (4, "c", "charcoal"),
    (5, "B", "black"),
    (6, "M", "magenta"),
    (7, "P", "pink"),
    (8, "R", "red"),
    (9, "b", "blue"),
    (10, "S", "sky blue"),
    (11, "Y", "yellow"),
    (12, "O", "orange"),
    (13, "r", "maroon"),
    (14, "N", "green"),
    (15, "p", "purple"),
]

# Labels the agent harvests from assistant text and carries to the next prompt.
NOTE_LABELS: list[tuple[str, str]] = [
    ("world", "World model"),
    ("goal", "Goal model"),
    ("action", "Action model"),
    ("findings", "Recent findings"),
    ("questions", "Open questions"),
    ("plan", "Plan"),
    ("cross_level", "Cross-level notes"),
]
LEVEL_SCOPED_NOTES = ("world", "goal", "action", "findings", "questions", "plan")


def _legend_text() -> str:
    return ", ".join(f"{char}={name} ({index})" for index, char, name in COLOR_LEGEND)


SYSTEM_PROMPT_BASE = f"""You are playing an unknown turn-based puzzle on a 64x64 grid of coloured cells. \
A game has several levels; finishing every level wins the game, and the score rewards using as few \
actions as possible on each level. Nobody tells you the rules: you discover them by acting and watching \
what changes.

You have one tool, `python`, which runs a code cell in a fresh restricted interpreter. Nothing you define \
survives between cells unless the harness says otherwise, so recompute what you need each time. Inside a \
cell the following names are ready to use:
- `current_frame` (alias `latest_frame`): the board now. Attributes: `.ascii` (64 text rows of 64 \
characters), `.step`, `.level`, `.shape`, and `.segmentation`, a dict with `nodes` (each has `id`, \
`color`, `hash`, `pixels`, `boundary` corner points as [row, col], `children`) and `adjacency_list`.
- `history`: every recorded entry, each with `.action` (text, empty for the opening entry) and `.frame`.
- `transitions`: one record per action taken, with `.action`, `.before_frame`, `.after_frame` and \
`.result`; `last_transition` is the newest one, and `previous_frame`, `last_action_frame`, \
`last_action` are shortcuts into it.
- `valid_actions` (labels you may use now) and `last_action_result` (a dict describing the latest move).
- `result`: assign anything here and it is returned to you with the cell output.
- `action(...)`: plays real moves. Pass one label, a dict like {{"action": "MOUSE", "row": r, "col": c}}, \
or a list of these to play several in order. It returns a result dict and refreshes every name above.

Colour characters in `.ascii`: {_legend_text()}.

Action labels: UP, DOWN, LEFT, RIGHT, SPACE, and MOUSE. MOUSE clicks one cell and needs integer \
`row` and `col` (0 to 63, row counted from the top). Only labels listed in `valid_actions` work.

How to look at the board:
- Start from `.segmentation`; it is far cheaper than printing the whole board. Print ASCII only for \
small cropped regions you need to inspect closely.
- After every move, compare the previous frame with the current one: list which cells changed and \
which objects moved, appeared or vanished. A move that changes nothing is information too.
- Read the scene as objects on a background. There may be no character you control at all; clicks \
or global moves may act on several things at once. Thin bars along the edges are often counters or \
timers that shrink as you act; do not mistake them for pieces to click.
- When the level number goes up the layout is new: drop assumptions tied to the old layout and \
look again. Reaching WIN means the whole game is over, not just one level.

Keep a short running model of the game in your replies, using these labelled lines, and revise it \
before you commit to moves:
World model: objects, their colours and how they behave.
Goal model: what seems to finish a level.
Action model: what each action label does.
Recent findings: what the latest moves showed.
Open questions: what you still need to test.
Plan: the next moves and why.
Cross-level notes: lessons that should survive into later levels.
The harness copies these lines into your next prompt.

Acting:
- Spend real moves deliberately: test one idea per move until an effect is understood.
- Once a sequence is reliable, send it as one list to `action` instead of one move per cell.
- Check `level_completed`, `game_over`, `run_complete` and `done` in each result and stop the cell \
when any is true; a batch already halts itself at such an event.

Tool use:
- Call the tool through the function-calling interface with a single `code` argument. Do not write \
tool-call tags or JSON imitations of a call in your reply text; such text is not executed.
- Keep printed output short, well under about 1024 tokens; long output is cut off.
"""

GATE_ADDENDUM_FORCED = """
Executable world model (required):
- Write a top-level `def predict(ascii, action)` that returns the board text expected after playing \
`action` (a label string such as "UP" or "MOUSE(row=3, col=5)") on the board text `ascii`.
- The harness keeps the top-level definitions of every successful cell that defines `predict` and \
places them before each later cell, so `predict` and its helpers stay available. A placeholder that \
returns the board unchanged is installed at the start; it can never pass.
- After each cell the harness replays your stored `predict` over this level's recorded transitions. \
Only moves that changed more than a couple of cells count, and the model is accepted once it gets at \
least 75% of at least 3 such moves right (a near miss of a cell or two is tolerated).
- On each level only a limited number of real actions is allowed before a model is accepted. After \
that, `action()` raises an error explaining the block until `predict` passes, and an oversized batch \
is refused with the number of moves still allowed.
"""

GATE_ADDENDUM_OPTIONAL = """
Optional world model:
- You may write a top-level `def predict(ascii, action)` returning the board text expected after \
playing `action` (a label string) on the board text `ascii`. Doing so is your choice; actions are \
never blocked.
- The top-level definitions of a successful cell that defines `predict` are kept and placed before \
each later cell. The harness replays `predict` over this level's recorded transitions and accepts it \
when it gets at least 75% of at least 3 informative moves right.
- `simulate(actions)` returns the predicted boards for a list of actions from the current board \
without spending real moves. It only works after an accepted `predict` also matched the next 3 \
informative real moves; until then it returns a message saying so.
"""


def system_prompt(gate_mode: str | None = None) -> str:
    """The system prompt; `gate_mode` is None, "forced" or "optional"."""
    if gate_mode == "forced":
        return SYSTEM_PROMPT_BASE + GATE_ADDENDUM_FORCED
    if gate_mode == "optional":
        return SYSTEM_PROMPT_BASE + GATE_ADDENDUM_OPTIONAL
    return SYSTEM_PROMPT_BASE


def render_notes(notes: dict[str, str]) -> str:
    lines = []
    for key, label in NOTE_LABELS:
        value = notes.get(key, "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _summary_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    count = summary.get("executed_count", 0)
    labels = ", ".join(summary.get("executed_actions") or []) or "none"
    parts = [
        (
            f"Last turn played {count} action(s): {labels} "
            f"(action #{summary.get('start_action_num')} to #{summary.get('end_action_num')})."
        ),
        (
            f"Afterwards: level {summary.get('level')}, state {summary.get('state')}, "
            f"board changed: {'yes' if summary.get('board_changed') else 'no'}."
        ),
    ]
    if summary.get("level_transition"):
        parts.append("The level changed; the board is a new layout.")
    if summary.get("run_complete"):
        parts.append("The game is complete.")
    if summary.get("game_over"):
        parts.append("That move ended in GAME_OVER; the level was restarted.")
    if summary.get("stop_reason"):
        parts.append(f"The batch stopped early: {summary['stop_reason']}.")
    return " ".join(parts)


RUNTIME_REMINDER = (
    "Reminder: each cell starts fresh with current_frame, history, transitions, last_transition, "
    "valid_actions, last_action_result and action() preloaded."
)


def user_prompt(
    *,
    step: int,
    level: int,
    number_of_levels: int | None,
    action_count: int,
    valid_actions: list[str],
    notes: dict[str, str],
    previous_summary: dict[str, Any] | None,
    first_turn: bool,
) -> str:
    level_text = f"{level} of {number_of_levels}" if number_of_levels else str(level)
    lines = [f"Turn {step}. Level {level_text}. Actions used so far: {action_count}."]
    summary = _summary_text(previous_summary)
    if summary:
        lines.append(summary)
    lines.append(f"Available actions: {', '.join(valid_actions) if valid_actions else 'none'}.")
    lines.append(RUNTIME_REMINDER)
    carried = render_notes(notes)
    if carried:
        lines.append("Your notes so far:\n" + carried)
    if first_turn:
        lines.append(
            "This is the opening turn. The image shows the current board. Survey it with python "
            "(segmentation first), write your first world model, then try a single informative action."
        )
    else:
        lines.append(
            "Look at what the last move changed, update your labelled notes, then choose the next "
            "action or a reliable batch."
        )
    return "\n\n".join(lines)


def corrective_prompt(markup_seen: bool) -> str:
    if markup_seen:
        return (
            "Your reply contained tool-call tags as plain text, and nothing was executed. Use the "
            "function-calling interface for the `python` tool, with the code in its `code` argument. "
            "Investigate with python, update your world model, then call action(...) from the cell."
        )
    return (
        "You have not taken an action yet. Investigate with python, update your world model, then "
        "call action(...) inside a python cell."
    )
