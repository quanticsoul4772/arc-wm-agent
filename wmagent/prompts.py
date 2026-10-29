"""Model-facing text for the agent (spec 2.4, T4 topic list).

Written fresh from the topic list; only functional identifiers (runtime globals,
action labels, note labels, `predict`/`simulate`) are shared with other harnesses.
The lengths are kept close to the original harness's prompts (system about 14k
characters with the forced gate, a typical user prompt about 2.9k), because
context eviction depends on prompt size (equivalence report F4).
"""
from __future__ import annotations

from typing import Any

# ASCII character -> palette index and color name (index order = palette order).
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
the rules, the objective or what the controls do. You find all of that out by acting and by reading \
the board carefully after every move.

== The game in more detail ==
- Every game is different. The same label can push a piece in one game, rotate a shape in another, and do \
nothing at all in a third, so treat each new game as unknown and learn its rules from what you observe.
- A level usually has one goal that stays fixed while you play it, such as bringing pieces into a target \
arrangement, filling or emptying a region, or reaching a marked place. Later levels tend to reuse the \
same ideas with more pieces, more obstacles or an extra twist.
- The score rewards efficiency: finishing a level in about as many actions as the reference solution is \
good, finishing faster is better, and every wasted move lowers the result. Not finishing a level scores \
nothing for that level and blocks all later ones, so first make sure you can finish, then make it short.
- Clicks and directions can both be available in one game. Check `valid_actions` on each turn, because the set \
of usable labels can change from one level to the next.
- A move can change many cells at once, change nothing, or change only a counter. All three outcomes are \
information about the rules.
- Colours tend to keep their meaning within a game, for example one for walls, one for the piece you move and \
one for targets, so a colour key you work out early keeps paying off in later turns.
- Some games show progress on the board itself, for example a target pattern that fills in or a piece count \
that drops; finding that signal early on makes every later decision much simpler.
- The level number, the step count and the actions you executed are reported in each turn's prompt, so you \
never need to count or track them yourself.

== Your tool ==
You have exactly one tool, `python`. Each call runs one code cell in a brand-new, locked-down interpreter. \
Variables, imports and functions from earlier cells are gone when the next cell starts (unless the section on \
the executable model below says the harness keeps something), so each cell recomputes what it needs. Only a \
short list of standard modules can be imported, grouped here by purpose: for numbers math, fractions, \
statistics, random and operator; for containers and iteration collections, itertools, heapq, bisect, \
functools and copy; for text re, string and json. There is no file or network access. A cell is stopped \
after 30 seconds, so avoid exhaustive searches over huge spaces.

== What a cell can see ==
These names are defined before your code runs:
- `current_frame` is the board right now (`latest_frame` is the same object). Its attributes:
    `.ascii`          the board as 64 text lines of 64 characters, one character per cell (legend below)
    `.segmentation`   the board split into connected single-color regions (described below)
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
- `action(...)` plays real moves in the game (see 'Playing moves').

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

== Seeing what a move did ==
After every move, compare the board before it with the board after it. Find the changed cells, then describe \
the change in terms of objects: which piece moved and how far, what appeared or vanished, what changed colour, \
and whether some counter grew or shrank. A move that changes nothing is also evidence: that action is blocked \
or meaningless in that position. Comparing the two segmentations (matching nodes by `hash`) tells you which \
objects moved and by how much; when many cells change, summarise them by colour pair and bounding box \
instead of listing every cell.

== Python habits that pay off ==
- Write small helpers inside a cell, for example one that returns the cells of a given colour, one that gives \
a bounding box, or one that finds a shape again after a move, and print compact summaries instead of raw data.
- Compute positions, counts and distances in code rather than estimating them from the picture.
- When choosing moves, rank the possible actions with a scoring function or explore sequences of moves \
in code, not by eye.
- IMPORTANT: when a goal needs several moves, above all for navigation, write an explicit search algorithm in \
code: BFS, DFS, flood fill, shortest-path or other pathfinding, beam search, or a bounded search over short \
action sequences. Do not plan long move sequences in your head.
- Print a few key numbers at the end of each cell so the next turn can build on them.
- Call python as many times as you need within a turn; do not ration calls while the state is still unclear.

== Visual cues ==
- Use the attached image as a second view of the same board: the outline of the scene, symmetry and colour \
groups are often easier to see there, and the text view then gives exact coordinates.
- Repeated shapes with one odd member, symmetric layouts with a gap, outlines waiting to be filled and \
matching colours placed far apart are frequent puzzle cues.
- Treat the board as a handful of objects on one or more background colours. The largest regions are usually \
background or walls; small distinct regions are the likely candidates for things that move or matter.
- Do not assume there is one piece you steer. Some games have no avatar at all: a direction may shift several \
objects at once, and a click may toggle, rotate or recolour whatever is under it, or something elsewhere.
- Bars or rows of small blocks along the outer edges of the board are often a move budget, a timer or a \
progress meter. They change as you act and can look like clickable pieces. DO NOT CLICK THE SEGMENTS OF \
THESE EDGE BARS: THEY ARE STATUS DISPLAYS, NOT PARTS OF THE PUZZLE.
- Look for relationships: a shape and a matching outline, a coloured key and a door of the same colour, a \
pattern in one corner and an unfinished copy of it elsewhere. Objectives are often 'make this match that'.

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
- One move: `action('LEFT')`
- One click: `action({'action': 'MOUSE', 'row': 12, 'col': 40})`
- Several moves in order: `action(['UP', 'UP', {'action': 'MOUSE', 'row': 5, 'col': 9}])`
`action` returns a result dict and refreshes `current_frame`, `history`, `transitions` and the other names in \
place, so code after the call already sees the new board. Useful result fields: `executed`, `action_num`, \
`level`, `score`, `state`, `board_changed`, `level_completed`, `game_over`, `run_complete`, `done`, \
`valid_actions`, and `error` when a request was rejected.

== How to work each turn ==
- Inspect the evidence with code, refine your working world model from the newest history, then search or \
score candidate actions and short sequences against your current goal before you act.
- Every executed action counts toward your score, so spend moves where your model says they help most.
- When a sequence is reliable, send it as one list in a single `action` call. You may also call `action` \
several times in one cell, for example to look at the board between two short sequences.
- A list halts by itself after a move that completes a level, wins the game or ends in GAME_OVER. In your own \
code, check `level_completed`, `game_over`, `run_complete` and `done` after each call and stop issuing moves \
as soon as any of them is true; look at the new board before planning further.
- If several moves in a row change nothing, stop and rethink instead of repeating them.
- Once you understand the state variables and what each action does, stop probing and search the inferred \
state space for a plan.
- When you are unsure which of two explanations is right, program a probe whose outcome tells them apart.
- Do not treat the objective as reaching an absolute row or column; use coordinates only to aim actions.

== Describing your understanding ==
State your current understanding of the game in your replies. These prefixes are optional helpers: Goal \
model, World model, Plan, Action model, Open questions, Recent findings, Cross-level notes.

== When a cell fails ==
An error inside a cell comes back as a short traceback with the line number in your code. Fix the cause and \
run the cell again; do not repeat a failing cell unchanged. Remember that names from earlier cells do not \
exist any more, which is the most common cause of a NameError.

== Tool calls ==
- Call `python` only through the function-calling interface, with your code in the `code` argument. Text in \
your reply that merely looks like a tool call (XML tags, JSON naming a function, or code you meant to run) is \
never executed.
- Your turn ends after a cell that plays at least one real move, so put inspection code before the \
`action(...)` call in the same cell when you want both.
- Keep printed output short, well under roughly 1,024 tokens per cell; longer output is cut off and wastes \
context. Summarise in code: count, group, and print only the lines that matter.
"""

SYSTEM_PROMPT_BASE = _SYSTEM_TEMPLATE.replace("{legend}", _legend_text())

GATE_ADDENDUM_FORCED = """
== Executable world model (required) ==
Besides your notes, keep a runnable model of the game: a top-level function `predict(ascii, action)`. Its \
`ascii` argument is a board in the same text form as `current_frame.ascii`, its `action` argument is a move \
label in the same form as the `.action` of a transition, and it returns the board text you expect after that \
move. A do-nothing placeholder stands in for it at the start of a game and must be replaced.
- When a cell that defines `predict` runs without an error, its top-level functions, classes, imports and \
constants are kept and placed in front of every later cell. Defining `predict` again replaces it.
- After each cell the stored `predict` is backtested on this level's recorded transitions in which the board \
changed. It is accepted once it reproduces at least 75% of them, with a tolerance of a few cells.
- On each level you may spend {probe_actions} real actions before a model is accepted; they are meant for \
learning what each action does. After that, `action()` is refused until `predict` is accepted and keeps \
predicting the new moves you make.
- Once your model is accepted, use it to search for a plan, for example breadth-first over predicted boards, \
before you spend real actions.
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


# WM_MECHANISMS m1a / all: the same section with the acceptance rule the gate enforces (the "base" text omits
# the move minimum and the change floor) and capture-on-deny (a predict in a cell whose action() is refused is
# still kept and judged).
GATE_ADDENDUM_FORCED_M1A = """
== Executable world model (required) ==
Besides your notes, keep a runnable model of the game: a top-level function `predict(ascii, action)`. Its \
`ascii` argument is a board in the same text form as `current_frame.ascii`, its `action` argument is a move \
label in the same form as the `.action` of a transition, and it returns the board text you expect after that \
move. A do-nothing placeholder stands in for it at the start of a game and must be replaced.
- When a cell that defines `predict` runs, its top-level functions, classes, imports and constants are kept \
and placed in front of every later cell, even if the cell's `action()` call was refused. Defining `predict` \
again replaces it. The tool result tells you whether it was accepted.
- After each cell the stored `predict` is backtested on this level's recorded moves that changed more than \
{change_floor} cells. It is accepted once there are at least {min_informative} such moves and it reproduces \
at least {pass_percent}% of them, with a tolerance of a few cells.
- On each level you may spend {probe_actions} real actions before a model is accepted; spend them on moves \
that change the board, so the backtest has enough informative moves. After that, `action()` is refused until \
`predict` is accepted and keeps predicting the new moves you make.
- Once your model is accepted, use it to search for a plan, for example breadth-first over predicted boards, \
before you spend real actions.
"""


def gate_addendum_forced(probe_actions: int, mechanisms: str = "base") -> str:
    """The required-model section, last in the system prompt: interface and rules only, no code."""
    if mechanisms == "base":
        return GATE_ADDENDUM_FORCED.replace("{probe_actions}", str(probe_actions))
    from wmagent.world_model_gate import CHANGE_FLOOR, MIN_INFORMATIVE, PASS_ACCURACY

    return (GATE_ADDENDUM_FORCED_M1A.replace("{probe_actions}", str(probe_actions))
            .replace("{change_floor}", str(CHANGE_FLOOR)).replace("{min_informative}", str(MIN_INFORMATIVE))
            .replace("{pass_percent}", str(int(PASS_ACCURACY * 100))))


def system_prompt(gate_mode: str | None = None, probe_actions: int = 8, mechanisms: str = "base") -> str:
    """The system prompt; `gate_mode` is None, "forced" or "optional"; `mechanisms` is the WM_MECHANISMS arm."""
    if gate_mode == "forced":
        return SYSTEM_PROMPT_BASE + gate_addendum_forced(probe_actions, mechanisms)
    if gate_mode == "optional":
        return SYSTEM_PROMPT_BASE + GATE_ADDENDUM_OPTIONAL
    return SYSTEM_PROMPT_BASE


def render_notes(notes: dict[str, str]) -> str:
    lines = []
    for key, label in NOTE_LABELS:
        value = notes.get(key, "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


MAX_LISTED_LABELS = 10

# Previous-turn facts, as the original conveys them (report round 6): the executed count and up to 10
# labels, exactly one level outcome, a game-over fact when one occurred. Nothing else (no stop reason,
# board change, state name or action-number span).
SUMMARY_COUNT = "Your last turn executed {count} {noun}: {labels}."
OUTCOME_RUN_COMPLETE = "That finished the whole game."
OUTCOME_NEW_LEVEL = "You completed one level; a new one."
OUTCOME_SAME_LEVEL = "You are still on the same level."
GAME_OVER_FACT = "It ended in GAME_OVER and the level was restarted."
# GAME_OVER while still on the same level: one combined sentence, as the original combines them.
SAME_LEVEL_GAME_OVER = "GAME_OVER; still on the same level, that restarted."
MOUSE_ARGUMENTS = "MOUSE needs integer row and col, each 0 to 63, top row 0."
NOTHING_EXECUTED = "No actions have been executed so far in this new game."
NOTHING_CAPTURED = "No previous action sequence was captured."


def _labels_text(labels: list[str]) -> str:
    if not labels:
        return "none"
    shown = ", ".join(labels[:MAX_LISTED_LABELS])
    if len(labels) > MAX_LISTED_LABELS:
        shown += ", ... (cut)"  # marker length matches the original's (report round 8)
    return shown


def _summary_text(summary: dict[str, Any]) -> str:
    count = int(summary.get("executed_count") or 0)
    parts = [SUMMARY_COUNT.format(count=count, noun="action" if count == 1 else "actions",
                                  labels=_labels_text(list(summary.get("executed_actions") or [])))]
    if summary.get("run_complete"):
        parts.append(OUTCOME_RUN_COMPLETE)
    elif summary.get("level_progressed"):
        parts.append(OUTCOME_NEW_LEVEL)
    elif summary.get("game_over"):
        return " ".join([*parts, SAME_LEVEL_GAME_OVER])
    else:
        parts.append(OUTCOME_SAME_LEVEL)
    if summary.get("game_over"):
        parts.append(GAME_OVER_FACT)
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

_FIRST_TURN_GUIDE = """This is the opening turn and no actions have been played yet. The attached \
image shows the current board. For this turn:
1. Inspect the board with python, starting from `current_frame.segmentation`: list the background colours, \
small objects, repeated shapes, and bars or counters along the edges.
2. Print ASCII only for small windows around objects you want to inspect closely.
3. Write a first world model, using any of the optional prefixes (Goal model, World model, Plan, Action \
model, Open questions, Recent findings, Cross-level notes) that help.
4. Choose an action or a short sequence that serves your current goal and play it with action(...) at the end \
of the cell.
5. After it runs, the result dict and the refreshed board are available in the same cell, so you can print a \
short summary of what changed before the cell ends.
6. Look at the whole board, not only the middle: counters, bars and small markers near the edges often tell \
you how many moves you have or how close you are to finishing, and they are easy to miss on a first look.\
 If the image shows what your text summary misses, say so, and look closer."""

_LATER_TURN_GUIDE = """For this turn:
1. Inspect the evidence with code: what the last moves changed, comparing `previous_frame` with \
`current_frame` and their segmentations, described as objects rather than raw cells. Separate changes to \
gameplay objects from changes that touch only counters, timers or other HUD elements.
2. Refine your working world model from the newest history; if the level changed, survey the new board \
before reusing old assumptions.
3. Search or score candidate actions and short sequences against your current goal, using your model \
wherever it helps.
4. Act: send a reliable short sequence as one batched call. You may call action(...) more than once in a \
cell, but stop as soon as a result reports level_completed, game_over, run_complete or done.\

5. Keep the reply focused: the code, the revised world model and the chosen moves matter more than long \
prose. If several moves in a row changed nothing, rethink the goal. Judge real progress against your goal, \
not only against the very last move."""


_TURN_REMINDERS = """Reminders:
- Your score on each level depends on how few actions you use, so spend moves where your world model and \
your goal say they will help.
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
    max_level_seen: int,
    valid_actions: list[str],
    notes: dict[str, str],
    previous_summary: dict[str, Any] | None,
) -> str:
    """`step` is the next step number (1 before anything ran); `max_level_seen` is the highest level any
    recorded frame showed and is mentioned only when it is above `level`."""
    header = f"Step {step}, level {level}."
    if max_level_seen > level:
        header += f" Highest level reached so far: {max_level_seen}."
    first_turn = previous_summary is None and step <= 1
    if previous_summary is not None:
        facts = _summary_text(previous_summary)
    else:
        facts = NOTHING_EXECUTED if first_turn else NOTHING_CAPTURED
    lines = [f"{header} {facts}"]
    labels_line = f"Labels accepted right now: {', '.join(valid_actions) if valid_actions else 'none'}."
    if "MOUSE" in valid_actions:  # only when clicks are possible, as in the original
        labels_line += " " + MOUSE_ARGUMENTS
    lines.append(labels_line + " Labels other than MOUSE take no arguments. The harness handles RESET after "
                 "GAME_OVER on its own.")
    lines.append(RUNTIME_REMINDER)
    carried = render_notes(notes)
    if not first_turn:
        # the revised-world-model mandate sits right before the carried notes, as in the original
        lines.append(WORLD_MODEL_MANDATE + ("\n" + NOTES_HEADER + "\n" + carried if carried else ""))
    lines.append(_FIRST_TURN_GUIDE if first_turn else _LATER_TURN_GUIDE)
    lines.append(_TURN_REMINDERS)
    return "\n\n".join(lines)


WORLD_MODEL_MANDATE = (
    "IMPORTANT: GIVE YOUR REVISED WORLD MODEL BEFORE YOU EXECUTE ANY NEW ACTION. By default, copy your "
    "previous world model and adjust it to the new evidence; the labelled prefixes are optional helpers."
)
NOTES_HEADER = (
    "Your carried notes from earlier replies, to copy forward and adjust: keep what still holds and fix what the "
    "newer board contradicts."
)


_CORRECTIVE_BODY = """Every turn has to end with at least one real move played from a python cell. \
Please continue like this:
1. Call the python tool to look at the current state. Start from current_frame.segmentation and from the \
difference between previous_frame and current_frame instead of printing the whole board.
2. Bring your labelled notes up to date with what you have learned so far, World model and Plan first, so \
that the next prompt carries them.
3. In the same cell or a following one, call action(...) with the move that best serves your goal, or with a \
list of moves if the sequence is already reliable. Inspecting without acting is fine in one cell, but a turn \
ends once a move is played.
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
