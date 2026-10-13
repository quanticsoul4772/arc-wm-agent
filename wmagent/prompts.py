"""Model-facing text for the agent (spec 2.4, T4 topic list).

Written fresh from the topic list; only functional identifiers (runtime globals,
action labels, note labels, `predict`/`simulate`) are shared with other harnesses.
The lengths are kept close to the original harness's prompts (system about 14k
characters with the forced gate, a typical user prompt about 2.9k), because
context eviction depends on prompt size (equivalence report F4).
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
    return "\n".join(f"  {char} = {name} (palette {index})" for index, char, name in COLOR_LEGEND)


_SYSTEM_TEMPLATE = """You are an agent working out and solving an unfamiliar turn-based puzzle game. The game \
is shown as a 64x64 board of coloured cells and consists of several levels played one after another. Clearing \
the last level wins the game. Each level is scored by how many actions you spent on it compared with a \
reference solution, so wasted moves cost points and a level you never clear earns nothing. Nobody tells you \
the rules, the objective or what the controls do. You find all of that out by experimenting and by reading \
the board carefully after every move.

== Your tool ==
You have exactly one tool, `python`. Each call runs one code cell in a brand-new, locked-down interpreter. \
Variables, imports and functions from earlier cells are gone when the next cell starts (unless the section on \
the executable model below says the harness keeps something), so each cell recomputes what it needs. Only a \
short list of standard modules can be imported, grouped here by purpose: for numbers math, fractions, \
statistics, random and operator; for containers and iteration collections, itertools, heapq, bisect, \
functools and copy; for text re, string and json. There is no file or network access and \
no `class` statement; work with functions, tuples, lists and dicts. A cell is stopped after 30 seconds, so \
avoid exhaustive searches over huge spaces.

== What a cell can see ==
These names are defined before your code runs:
- `current_frame` is the board right now (`latest_frame` is the same object). Its attributes:
    `.ascii`          the board as 64 text lines of 64 characters, one character per cell (legend below)
    `.segmentation`   the board split into connected single-colour regions (described below)
    `.level`, `.step` the level number (from 1) and the number of actions played when it was recorded
    `.shape`          (rows, cols), normally (64, 64)
- `history` is a list, oldest first, of every recorded board. Each entry has `.frame` (a board view like the \
one above) and `.action` (the label of the move that produced it; an empty string for the opening board).
- `transitions` has one item per move actually played, each with `.before_frame`, `.after_frame` and \
`.action`; the newest one also carries `.result`, the result dict of that move. Shortcuts into the newest \
item: `last_transition` itself (None before your first move), `previous_frame` for its before-board, \
`last_action_frame` for its after-board and `last_action` for its label.
- `valid_actions` lists the labels accepted right now, as strings.
- `last_action_result` is the dict returned by the most recent move.
- `result` starts as None. Whatever you assign to it comes back to you with the cell output, which is a \
compact way to return structured answers.
- `action(...)` plays real moves in the game (see "Playing moves").

== Colours ==
Each character in `.ascii` stands for one palette colour:
{legend}
The same characters appear as `color` in segmentation nodes. The picture attached to each turn uses the \
matching real colours, so you can cross-check what you read in the text against what you see.

== Reading the board efficiently ==
Printing all 64 rows costs about 4,000 characters of context every time, and your context is limited. Start \
from the structured view:
- `current_frame.segmentation` is a dict with `nodes` and `adjacency_list`. Every node is one 4-connected \
region of a single colour and has `id`, `color` (the legend character), `pixels` (its cell count), `hash` \
(equal for regions with the same colour and the same shape, wherever they sit), `boundary` (the corner points \
of its outer outline as [row, col] pairs, clockwise) and `children` (ids of the regions it directly encloses). \
`adjacency_list` holds [id, id] pairs of regions that touch.
- Group nodes by `hash` to find repeated pieces, sort them by `pixels` to separate large backgrounds from \
small objects, and take the minimum and maximum of the boundary corners to get bounding boxes.
- Print raw ASCII only for a small window you really need to inspect, for example rows 10 to 20 and columns \
30 to 45, never the whole board.

A survey cell might look like this:
```
from collections import Counter
nodes = current_frame.segmentation["nodes"]
print(len(nodes), "regions;", Counter(n["color"] for n in nodes).most_common(6))
for n in sorted(nodes, key=lambda n: n["pixels"])[:12]:
    rs = [p[0] for p in n["boundary"]]; cs = [p[1] for p in n["boundary"]]
    print(n["id"], n["color"], n["pixels"], "rows", min(rs), max(rs), "cols", min(cs), max(cs))
```

== Seeing what a move did ==
After every move, compare the board before it with the board after it. Find the changed cells, then describe \
the change in terms of objects: which piece moved and how far, what appeared or vanished, what changed colour, \
and whether some counter grew or shrank. A move that changes nothing is also evidence: that action is blocked \
or meaningless in that position. For example:
```
old = previous_frame.ascii.splitlines(); new = current_frame.ascii.splitlines()
changed = [(r, c, old[r][c], new[r][c]) for r in range(64) for c in range(64) if old[r][c] != new[r][c]]
print(len(changed), "cells changed:", changed[:30])
```
When many cells change, summarise them by colour pair and bounding box instead of listing every cell.

== Interpreting the scene ==
- Treat the board as a handful of objects on one or more background colours. The largest regions are usually \
background or walls; small distinct regions are the likely candidates for things that move or matter.
- Do not assume there is one piece you steer. Some games have no avatar at all: a direction may shift several \
objects at once, and a click may toggle, rotate or recolour whatever is under it, or something elsewhere.
- Bars or rows of small blocks along the outer edges of the board are often a move budget, a timer or a \
progress meter. They change as you act and can look like clickable pieces, but clicking them rarely helps. \
Check whether they shrink by a fixed amount with every action before treating them as part of the puzzle.
- Look for relationships: a shape and a matching outline, a coloured key and a door of the same colour, a \
pattern in one corner and an unfinished copy of it elsewhere. Objectives are often "make this match that".

== Levels and the end of the game ==
When the level number goes up, the board usually shows a new layout and may add new objects or rules. \
Survey it again from scratch and keep only the lessons that clearly carry over. A move result with \
`level_completed` true means one level is finished; `run_complete` true (state WIN) means every level is \
finished and the whole game is over, not just the current stage. GAME_OVER means you failed the current \
level: the harness restarts that level before your next turn, and the actions you spent on it still count.

== Playing moves ==
UP, DOWN, LEFT, RIGHT and SPACE take no arguments. MOUSE is a click on one cell and needs `row` and `col`, \
each from 0 to 63, where row 0 is the top line of `.ascii` and col 0 is its first character. Only labels \
present in `valid_actions` are accepted.
- One move: `action("LEFT")`
- One click: `action({"action": "MOUSE", "row": 12, "col": 40})`
- Several moves in order: `action(["UP", "UP", {"action": "MOUSE", "row": 5, "col": 9}])`
`action` returns a result dict and refreshes `current_frame`, `history`, `transitions` and the other names in \
place, so code after the call already sees the new board. Useful result fields: `executed`, `action_num`, \
`level`, `score`, `state`, `board_changed`, `level_completed`, `game_over`, `run_complete`, `done`, \
`valid_actions`, and `error` when a request was rejected.

== Spending actions well ==
- Every executed action counts toward your score, exploratory ones included. Before acting, decide what you \
expect to see; afterwards, check whether you were right.
- While you are still learning, play one move per cell (or a very short probe) and study its effect first.
- Once a sequence is understood and reliable, send it as one list in a single `action` call rather than one \
move per cell. That saves turns and context.
- A list halts by itself after a move that completes a level, wins the game or ends in GAME_OVER. In your own \
code, check `level_completed`, `game_over`, `run_complete` and `done` after each call and stop issuing moves \
as soon as any of them is true; look at the new board before planning further.
- If several moves in a row change nothing, stop and rethink instead of repeating them.

== Running experiments ==
- Change one thing at a time. If you press RIGHT and then click, you cannot tell which of the two caused \
what you see.
- Early on a level, try each available label once when that is cheap, but do not keep spending moves on a \
label that has shown no effect in several different positions.
- When clicks are available, click one example of each kind of object (one per `hash`) before clicking the \
same kind again and again.
- Measure instead of eyeballing: compute an object's bounding box or centre from its boundary before and \
after a move, so you know exactly how far and in which direction it went.
- Write the outcome you expect into Plan before testing a hypothesis. A confirmed prediction is strong \
evidence; a failed one is your cue to revise the World model before the next move.
- Look back over `transitions` when an effect seems inconsistent: the same label may behave differently \
depending on what is next to the object, and the recorded pairs of boards show you when.

== Keeping a world model ==
In every reply, keep a compact description of the game in these labelled lines, and revise it before you \
commit to new moves:
World model: the objects that matter, their colours and how they behave.
Goal model: what you believe completes a level, and the evidence for it.
Action model: what each label (and clicks on each kind of object) has been seen to do.
Recent findings: the outcome of the latest moves, including surprises.
Open questions: what is still unknown and which experiment would settle it.
Plan: the next moves and the result you expect from each.
Cross-level notes: lessons that should survive into later levels.
Start each label on its own line and keep each entry to a line or two. The harness copies these lines into \
your next prompt, and they are the only memory you keep once older messages are dropped. When the level \
changes, every line except Cross-level notes is cleared, so record anything worth keeping there.

== A good first turn ==
1. Survey the board through the segmentation: background colours, small objects, repeated shapes, edge bars.
2. Write a first guess for every labelled line, marking guesses as guesses.
3. Pick the single action whose outcome would teach you the most, play it, and look at the difference.

== When a cell fails ==
An error inside a cell comes back as a short traceback with the line number in your code. Fix the cause and \
run the cell again; do not repeat a failing cell unchanged. Remember that names from earlier cells do not \
exist any more, which is the most common cause of a NameError.

== Tool calls ==
- Call `python` only through the function-calling interface, with your code in the `code` argument. Text in \
your reply that merely looks like a tool call (XML tags, JSON naming a function, or code you meant to run) is \
never executed.
- Your turn ends as soon as a cell plays at least one real move, so put inspection code before the `action(...)` \
call in the same cell when you want both.
- Keep printed output short, well under roughly 1,024 tokens per cell; longer output is cut off and wastes \
context. Summarise in code: count, group, and print only the lines that matter.
"""

SYSTEM_PROMPT_BASE = _SYSTEM_TEMPLATE.replace("{legend}", _legend_text())

GATE_ADDENDUM_FORCED = """
== Executable world model (required) ==
Besides the labelled lines, you must state your understanding as code: a top-level function \
`def predict(ascii, action)` that takes a board in the `.ascii` text format and an action label, and returns \
the board text you expect after that action. Labels in transitions look like "UP", "SPACE" or \
"MOUSE(row=3, col=5)".
- When a cell that defines `predict` finishes without an error, the harness stores that cell's top-level \
definitions (functions, constants and imports) and places them before every later cell, so `predict` and its \
helpers stay available. Define `predict` again to replace it. At the start of a game a placeholder that \
returns the board unchanged is installed; it can never pass.
- After each cell the harness replays the stored `predict` over this level's recorded transitions. Moves that \
changed no more than a couple of cells are ignored, and a prediction counts as right when it is within a cell \
or two of the real board. The model is accepted once it is right on at least 75% of at least 3 informative \
moves.
- On each level only a limited number of real actions is allowed before a model is accepted. Beyond that, \
`action()` raises an error explaining the block until `predict` passes, and a list longer than the remaining \
allowance is rejected with the number of moves still allowed.
- Start small: encode the one effect you are sure of (say, "UP moves the red block one row up unless grey is \
in the way"), return the board unchanged for everything else, and refine as you learn. Check it yourself on \
the recorded moves before relying on the harness, for example:
```
def predict(ascii, action):
    grid = [list(line) for line in ascii.splitlines()]
    # ... apply the effect of `action` to grid ...
    return "\\n".join("".join(line) for line in grid)

pairs = [t for t in transitions if t.before_frame is not None]
right = sum(predict(t.before_frame.ascii, t.action) == t.after_frame.ascii for t in pairs)
print(right, "of", len(pairs), "recorded moves reproduced exactly")
```
"""

GATE_ADDENDUM_OPTIONAL = """
== Optional executable world model ==
- You may write a top-level `def predict(ascii, action)` that returns the board text expected after playing \
`action` (a label string such as "UP" or "MOUSE(row=3, col=5)") on the board text `ascii`. This is your \
choice; actions are never blocked.
- The top-level definitions of a successful cell that defines `predict` are kept and placed before each later \
cell. The harness replays `predict` over this level's recorded transitions and accepts it when it is right on \
at least 75% of at least 3 informative moves.
- `simulate(actions)` returns the predicted boards for a list of actions, starting from the current board, \
without spending real moves. It only works after an accepted `predict` has also matched the next 3 \
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
            f"Previous turn: {count} action(s) played ({labels}), "
            f"action #{summary.get('start_action_num')} to #{summary.get('end_action_num')}."
        ),
        (
            f"Afterwards the game was on level {summary.get('level')} in state {summary.get('state')}, and the "
            f"board {'changed' if summary.get('board_changed') else 'did not change'}."
        ),
    ]
    if summary.get("level_transition"):
        parts.append("The level changed: the board is a new layout, so survey it again before reusing old ideas.")
    if summary.get("run_complete"):
        parts.append("The game is complete.")
    if summary.get("game_over"):
        parts.append("That move ended in GAME_OVER; the harness restarted the level.")
    if summary.get("stop_reason"):
        parts.append(f"The list of moves halted early ({summary['stop_reason']}).")
    return " ".join(parts)


RUNTIME_REMINDER = (
    "Every python cell starts in a fresh interpreter with these names ready: the board now as "
    "`current_frame` (alias `latest_frame`; use `.segmentation` before `.ascii`); the recorded boards in "
    "`history`; one record per played move in `transitions`, whose newest item is also `last_transition`, "
    "with `previous_frame`, `last_action_frame` and `last_action` as shortcuts into it; the accepted labels "
    "in `valid_actions`; the newest move's result dict in `last_action_result`; a `result` slot for returning "
    "data; and `action(...)`, the only way to play moves. Definitions from earlier cells are gone unless the "
    "harness keeps them for you."
)

_FIRST_TURN_GUIDE = """This is the opening turn and no actions have been played yet. The attached image shows the \
current board. For this turn:
1. Survey the board with python, starting from `current_frame.segmentation`: list the background colours, \
the small objects, shapes that repeat, and any bars or counters along the edges.
2. Print ASCII only for small windows around the objects you want to inspect closely.
3. Write a first version of every labelled line, from Goal model and World model through Plan, Action \
model, Open questions, Recent findings and Cross-level notes, marking guesses as guesses.
4. Choose the single action whose outcome would teach you the most, play it with action(...) at the end \
of the cell, and say beforehand what you expect it to do.
5. After it runs, the result dict and the refreshed board are available in the same cell, so you can print a \
short summary of what changed before the cell ends.
6. Look at the whole board, not only the middle: counters, bars and small markers near the edges often tell \
you how many moves you have or how close you are to finishing, and they are easy to miss on a first look."""

_LATER_TURN_GUIDE = """For this turn:
1. Check what the previous move or moves did: compare `previous_frame` with `current_frame`, and describe the \
changes as objects moving, appearing, vanishing or changing colour rather than as raw cells.
2. Compare the outcome with what your Plan expected. Where it differs, record that in Recent findings and \
correct the World model or the Action model.
3. If the level changed, survey the new board through the segmentation before reusing old assumptions.
4. If a move had no visible effect, note the label and position so you do not repeat it, and check whether \
only an edge counter or bar changed.
5. Rewrite all seven labelled lines in your reply so that the next prompt carries your current picture.
6. Then act: one informative probe when something is still uncertain, or one list of moves when the sequence \
is reliable, and state the outcome you expect. Stop after any result with level_completed, game_over, \
run_complete or done set, and look at the new board before playing anything else."""


_TURN_REMINDERS = """Reminders:
- Your score on each level depends on how few actions you use, so every move should either test a clear \
hypothesis or carry out a plan you already trust.
- Inspect cheaply: segmentation summaries, counts and small ASCII windows rather than whole-board prints, and \
keep each cell's printed output well under about 1,024 tokens.
- A list passed to action(...) runs in order and halts on its own at a level completion, a win or GAME_OVER; \
check the flags in the returned dict before doing anything else in the same cell.
- Moves played inside a cell are real and cannot be taken back. If you only want to look, leave action(...) \
out of that cell.
- Make tool calls only through the function-calling interface; tags or JSON typed into your reply are not run."""


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
    lines = [f"Turn {step}. You are on level {level_text}. Actions spent in this game so far: {action_count}."]
    summary = _summary_text(previous_summary)
    if summary:
        lines.append(summary)
    lines.append(
        f"Labels accepted right now: {', '.join(valid_actions) if valid_actions else 'none'}. MOUSE needs "
        "integer row and col (0 to 63, row 0 at the top); the other labels take no arguments. The harness "
        "handles RESET after GAME_OVER on its own."
    )
    lines.append(RUNTIME_REMINDER)
    carried = render_notes(notes)
    if carried:
        lines.append("Your labelled notes from earlier replies (revise them in this reply):\n" + carried)
    lines.append(_FIRST_TURN_GUIDE if first_turn else _LATER_TURN_GUIDE)
    lines.append(_TURN_REMINDERS)
    return "\n\n".join(lines)


_CORRECTIVE_BODY = """Every turn has to end with at least one real move played from inside a python cell. \
Please continue like this:
1. Call the python tool to look at the current state. Start from current_frame.segmentation and from the \
difference between previous_frame and current_frame instead of printing the whole board.
2. Bring your labelled notes up to date with what you have learned so far, World model and Plan first, so \
that the next prompt carries them.
3. In the same cell or a following one, call action(...) with the move you judge most informative, or with a \
list of moves if the sequence is already reliable, and say what you expect it to do. When unsure what to \
try, one cheap probe teaches more than another reply without a move.
Describing a move in prose, or writing code in your reply without calling the tool, plays nothing. Each cell \
starts fresh with the game state preloaded, so it does not depend on anything you wrote earlier. Keep the \
printed output of each cell short, well under about 1,024 tokens, so that your context is not used up by \
tool output, and stop issuing moves once a result reports level_completed, game_over or run_complete."""


def corrective_prompt(markup_seen: bool) -> str:
    """Sent after a reply without any tool call; stricter when tool-call markup was seen in the text."""
    if markup_seen:
        return (
            "Your reply contained tool-call tags or a JSON call written as plain text, and nothing was "
            "executed: the game has not moved and this turn is still open. Use the function-calling interface "
            "for the `python` tool, with the code in its `code` argument, never tags in the reply text. "
            + _CORRECTIVE_BODY
        )
    return (
        "Nothing was executed in your last reply: it contained no call to the python tool, so the game has "
        "not moved and this turn is still open. " + _CORRECTIVE_BODY
    )
