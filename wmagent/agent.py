"""Tool-using agent loop (spec 2.4): chat requests to an OpenAI-compatible server,
python tool cells in the sandbox, context eviction, carried notes and the
world-model gate hooks."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from wmagent import prompts
from wmagent.world_model_gate import BACKTEST_DRIVER, WorldModelGate

# ---------------------------------------------------------------------------
# configuration (read from the environment at import time; names are the interface)
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_num(name: str, default: float, cast: Callable[[str], Any] = float) -> Any:
    try:
        return cast(_env(name, str(default)))
    except ValueError:
        return cast(str(default))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AgentConfig:
    base_url: str
    model: str
    provider: str
    context_window: int
    max_output: int
    tool_steps: int
    tool_timeout: float
    tool_output_tokens: int
    yield_seconds: float
    temperature: float
    top_p: float
    top_k: int
    seed: int
    enable_thinking: bool
    multimodal_context: str
    multimodal_upscale: int

    @classmethod
    def from_env(cls) -> AgentConfig:
        model = _env("LOCAL_ANALYZER_MODEL_ID", "vrfai/Qwen3.6-27B-FP8")
        base, provider = resolve_endpoint(model, os.environ)
        return cls(
            base_url=base,
            model=model,
            provider=provider,
            context_window=_env_num("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768, int),
            max_output=_env_num("LOCAL_ANALYZER_MAX_OUTPUT", 0, int),
            tool_steps=_env_num("LOCAL_ANALYZER_TOOL_STEPS", 0, int),
            tool_timeout=min(30.0, _env_num("LOCAL_ANALYZER_TOOL_TIMEOUT", 30.0)),
            tool_output_tokens=_env_num("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024, int),
            yield_seconds=_env_num("LOCAL_ANALYZER_YIELD_SECONDS", 60.0),
            temperature=_env_num("LOCAL_ANALYZER_TEMPERATURE", 0.6),
            top_p=_env_num("LOCAL_ANALYZER_TOP_P", 0.95),
            top_k=_env_num("LOCAL_ANALYZER_TOP_K", 20, int),
            seed=_env_num("LOCAL_ANALYZER_SEED", -1, int),
            enable_thinking=_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True),
            multimodal_context=_env("MULTIMODAL_CONTEXT", "current_grid"),
            multimodal_upscale=_env_num("MULTIMODAL_UPSCALE", 4, int),
        )

    @property
    def reply_reserve(self) -> int:
        return self.max_output if self.max_output > 0 else 512

    @property
    def budget(self) -> int:
        return self.context_window - self.reply_reserve - SAFETY_TOKENS

    @property
    def tool_output_chars(self) -> int:
        return tool_output_limit(self.tool_output_tokens)


LOCAL_MODEL_PRESETS = frozenset({"local", "local-qwen", "qwen-local", "qwen"})
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def resolve_endpoint(model: str, env: Any) -> tuple[str, str]:
    """Base URL and provider (behaviour note F20). A local preset model id reads the LOCAL_ANALYZER_*
    variables; any other id prefers OPENAI_BASE_URL whenever it exists (even if empty) and OPENAI_PROVIDER."""
    if model.strip().lower() in LOCAL_MODEL_PRESETS:
        # set-but-empty is taken as is (and rejected at construction), as in the original
        base = env.get("LOCAL_ANALYZER_BASE_URL", DEFAULT_BASE_URL)
        provider = env.get("LOCAL_ANALYZER_PROVIDER") or env.get("OPENAI_PROVIDER") or "vllm"
    else:
        base = env["OPENAI_BASE_URL"] if "OPENAI_BASE_URL" in env else (
            env.get("LOCAL_ANALYZER_BASE_URL") or DEFAULT_BASE_URL)
        provider = env.get("OPENAI_PROVIDER") or env.get("LOCAL_ANALYZER_PROVIDER") or "vllm"
    return _rewrite_docker_host(base.strip()).rstrip("/"), provider.strip()


def _rewrite_docker_host(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if parts.hostname != "host.docker.internal":
        return url
    netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


SAFETY_TOKENS = 512
MAX_PERSISTED_ASSISTANT = 30
CONFIG = AgentConfig.from_env()

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        # Length kept close to the original harness's tool definition (equivalence report F4): the
        # estimator counts the tools JSON in every request.
        "description": "Run one Python cell in a fresh, restricted interpreter that starts with the game state "
        "already loaded. Nothing survives from earlier cells except definitions the harness keeps for you. "
        "Ready names: current_frame (alias latest_frame) with .ascii, .segmentation, .level, .step and .shape; "
        "history entries with .action and .frame; transitions with .before_frame, .after_frame, .action and "
        ".result, plus the shortcuts last_transition, previous_frame, last_action_frame and last_action; "
        "valid_actions; last_action_result; and result, which is returned to you when the cell ends. Call "
        "action('UP'), action({'action': 'MOUSE', 'row': r, 'col': c}) or action([...]) inside the cell to play "
        "real moves; each call returns a result dict and refreshes the names above. The cell is stopped after 30 seconds, and printed output beyond about 4,000 "
        "characters is cut off.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source for the cell. Print short "
                                    "summaries or assign to `result`; a cell that plays a move ends the turn."}},
            "required": ["code"],
        },
    },
}
TOOLS = [PYTHON_TOOL]
TOOL_CHOICE = "auto"

# Model-facing labels for engine action names (spec 2.3).
ENGINE_TO_LABEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}

ASCII_ALPHABET = "WwgGcBMPRbSYOrNp"
PALETTE_RGB = [
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
]

CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "reduce the length of the input prompt",
    "parameter=input_tokens",
    '"param":"input_tokens"',
)

# compact_result key rules (equivalence report, behaviour note F11), in emission order.
COMPACT_COPIED_KEYS = ("action_num", "level", "score", "reward", "state")
COMPACT_FLAG_KEYS = ("board_changed", "done", "level_completed", "game_over", "run_complete")
COMPACT_BASE_KEYS = ("executed", *COMPACT_COPIED_KEYS, "valid_actions", *COMPACT_FLAG_KEYS, "action_display")
COMPACT_KEYS = (*COMPACT_BASE_KEYS, "executed_actions")
TIMING_KEYS = ("run_elapsed_seconds", "time_remaining_seconds")
# Terminal reasons that arm the per-cell latch, in priority order.
TERMINAL_FLAGS = ("run_complete", "game_over", "level_completed", "done")


class AnalyzerRequestError(RuntimeError):
    """A chat request failed (HTTP error, transport error or unreadable body)."""


class ContextLengthError(AnalyzerRequestError):
    """The server rejected the prompt as too long."""


def is_context_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in CONTEXT_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# helpers: text cleanup, markup recovery, notes, eviction
# ---------------------------------------------------------------------------

_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def clean_text(text: Any) -> str:
    """Drop <think> tags and blank lines."""
    if not text:
        return ""
    stripped = _THINK_TAG.sub("", str(text))
    return "\n".join(line for line in stripped.splitlines() if line.strip())


MARKUP_DETECT = re.compile(r"<tool_call|<function=", re.IGNORECASE)
# Only complete blocks are recovered and stripped (behaviour note F9); incomplete markup stays in the text.
_COMPLETE_CALL = re.compile(
    r"<tool_call>\s*<function=([^>]*)>(.*?)</function>\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_PARAM = re.compile(r"<parameter=([^>]*)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)


def recover_tool_calls(text: str) -> list[dict[str, Any]]:
    """Tool calls written as complete qwen3-coder XML blocks, in order of appearance."""
    return [
        {"name": name.strip(), "arguments": {key.strip(): value.strip() for key, value in _PARAM.findall(body)}}
        for name, body in _COMPLETE_CALL.findall(text or "")
    ]


def _dedup_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for call in calls:
        key = json.dumps([call.get("name"), call.get("arguments")], sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def strip_markup(text: str) -> str:
    return clean_text(_COMPLETE_CALL.sub("", text or "")).strip()


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Label text -> note key, primaries first, then the fallbacks (behaviour note F17).
_PRIMARY_LABELS = [(label, key) for key, label in prompts.NOTE_LABELS]
_FALLBACK_LABELS = [("Hypothesis", "world"), ("History check", "findings"), ("Next test", "plan")]


def _strip_bullets(line: str) -> str:
    line = line.strip()
    while line[:1] in ("-", "*"):
        line = line[1:].lstrip()
    return line


def extract_notes(text: str) -> dict[str, str]:
    """Labelled notes from assistant content. A label line starts (after `-`/`*` bullets) with the label
    immediately followed by `:`; other non-blank lines continue the current label as written (trimmed only),
    blank lines are skipped,
    a repeated label appends, and each note is collapsed to one line. A fallback label fills its key only
    when the primary label produced nothing."""
    pieces: dict[str, list[str]] = {}
    current: str | None = None
    for raw in (text or "").splitlines():
        written = raw.strip()
        if not written:
            continue
        # bullets are stripped only to test for a label line; a continuation line is kept as written
        line = _strip_bullets(written)
        lowered = line.lower()
        for label, _key in (*_PRIMARY_LABELS, *_FALLBACK_LABELS):
            if lowered.startswith(label.lower() + ":"):
                current = label
                rest = line[len(label) + 1:].strip()
                pieces.setdefault(label, [])
                if rest:
                    pieces[label].append(rest)
                break
        else:
            if current is not None:
                pieces[current].append(written)
    values = {label: " ".join(" ".join(parts).split()) for label, parts in pieces.items()}
    notes = {key: values[label] for label, key in _PRIMARY_LABELS if values.get(label)}
    for label, key in _FALLBACK_LABELS:
        if values.get(label) and key not in notes:
            notes[key] = values[label]
    return notes


def estimate_tokens(messages: list[dict[str, Any]]) -> float:
    payload = {"messages": messages, "tools": TOOLS, "tool_choice": TOOL_CHOICE}
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=True)) / 3


def drop_oldest_block(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the oldest block from history (system excluded); never the newest message."""
    if len(history) <= 1:
        return history
    limit = len(history) - 1
    end = 1
    if history[0].get("role") == "user":
        while end < limit and history[end].get("role") == "tool":
            end += 1
        while end < limit and history[end].get("role") != "user":
            end += 1
    else:
        while end < limit and history[end].get("role") == "tool":
            end += 1
    return history[end:]


def _user_first(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = 0
    while start < len(history) - 1 and history[start].get("role") != "user":
        start += 1
    return history[start:]


def trim_messages(messages: list[dict[str, Any]], budget: float) -> list[dict[str, Any]]:
    """Drop oldest history blocks until under budget, then start history at a user message."""
    system, history = messages[0], list(messages[1:])
    while len(history) > 1 and estimate_tokens([system, *history]) > budget:
        history = drop_oldest_block(history)
    return [system, *_user_first(history)]


def persistable_history(messages: list[dict[str, Any]], budget: float | None = None) -> list[dict[str, Any]]:
    """History to keep after a turn: trimmed to the budget, no system, the last 30 assistant
    messages. If the cut lands right after a user message, that message is kept too; otherwise
    leading messages are dropped up to the next user message."""
    if budget is not None and messages and messages[0].get("role") == "system":
        messages = trim_messages(messages, budget)
    history = [m for m in messages if m.get("role") != "system"]
    assistant_positions = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
    if len(assistant_positions) > MAX_PERSISTED_ASSISTANT:
        cut = assistant_positions[-MAX_PERSISTED_ASSISTANT]
        if cut > 0 and history[cut - 1].get("role") == "user":
            cut -= 1
        history = history[cut:]
    while history and history[0].get("role") != "user":
        history = history[1:]
    return history


# ---------------------------------------------------------------------------
# frames, images, sandbox state
# ---------------------------------------------------------------------------


def grid_to_ascii(grid: list[list[int]]) -> str:
    top = len(ASCII_ALPHABET) - 1
    return "\n".join("".join(ASCII_ALPHABET[min(max(int(v), 0), top)] for v in row) for row in grid)


def grid_png_data_url(grid: list[list[int]], upscale: int) -> str:
    from PIL import Image

    height, width = len(grid), len(grid[0]) if grid else 0
    image = Image.new("RGB", (width, height))
    top = len(PALETTE_RGB) - 1
    image.putdata([PALETTE_RGB[min(max(int(v), 0), top)] for row in grid for v in row])
    if upscale > 1:
        image = image.resize((width * upscale, height * upscale), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _frame_view(frame: dict[str, Any]) -> dict[str, Any]:
    grid = frame.get("grid") or []
    return {
        "ascii": grid_to_ascii(grid),
        "step": frame.get("step", 0),
        "level": max(1, int(frame.get("level") or 1)),
        "shape": [len(grid), len(grid[0]) if grid else 0],
        "grid": grid,
    }


def load_runtime_state(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _current_level(state_path: Path) -> int:
    frame = load_runtime_state(state_path).get("current_frame") or {}
    return max(1, int(frame.get("level") or 1))


def sandbox_state(runtime: dict[str, Any], valid_actions: list[str], last_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_frame": _frame_view(runtime.get("current_frame") or {}),
        "history": [
            {"action": entry.get("action") or "", "frame": _frame_view(entry.get("frame") or {})}
            for entry in runtime.get("history") or []
        ],
        "valid_actions": list(valid_actions),
        "last_action_result": dict(last_result or {}),
    }


def to_label(name: str) -> str:
    return ENGINE_TO_LABEL.get(str(name).upper(), str(name))


def compact_result(payload: dict[str, Any]) -> dict[str, Any]:
    """The step_env payload reduced to what the model sees (behaviour note F11)."""
    display = payload.get("action_display") or payload.get("action_name") or None
    compact: dict[str, Any] = {"executed": bool(payload.get("executed"))}
    compact.update({key: payload.get(key) for key in COMPACT_COPIED_KEYS})
    compact["valid_actions"] = payload.get("valid_actions", [])
    compact.update({key: bool(payload.get(key)) for key in COMPACT_FLAG_KEYS})
    compact["action_display"] = display
    executed_actions = payload.get("executed_actions")
    if isinstance(executed_actions, list) and executed_actions:
        compact["executed_actions"] = [str(item).strip() for item in executed_actions if str(item).strip()]
    elif display:
        compact["executed_actions"] = [display]
    counts = [payload[key] for key in ("requested_count", "executed_count") if payload.get(key) is not None]
    batch_size = counts[0] if counts else 1
    if batch_size > 1 or payload.get("stopped_early"):
        compact["requested_count"] = payload.get("requested_count", batch_size)
        compact["executed_count"] = payload.get("executed_count", batch_size)
        compact["stopped_early"] = bool(payload.get("stopped_early"))
    for key in ("stop_reason", "stop_detail"):
        if payload.get(key):
            compact[key] = payload[key]
    compact.update({key: payload[key] for key in TIMING_KEYS if key in payload})
    if payload.get("error"):
        compact["error"] = payload["error"]
    return compact


def latch_stub(latched: dict[str, Any], reason: str, requested: int) -> dict[str, Any]:
    """What a later action() in the same cell returns after a terminal result (behaviour note F19)."""
    details = {
        "run_complete": "The game is already won; nothing more is played.",
        "game_over": "An earlier action() in this cell ended in GAME_OVER; the level restarts before your next turn.",
        "level_completed": "An earlier action() in this cell completed the level; look at the new board before "
        "playing again.",
        "done": "An earlier action() in this cell finished the run; nothing more is played.",
    }
    return {
        "executed": False,
        "action_num": latched.get("action_num"),
        "level": latched.get("level"),
        "score": latched.get("score"),
        "reward": 0.0,
        "state": latched.get("state"),
        "valid_actions": [],
        "board_changed": False,
        **{key: bool(latched.get(key)) for key in ("done", "level_completed", "game_over", "run_complete")},
        "requested_count": requested,
        "executed_count": 0,
        "stopped_early": True,
        "stop_reason": f"previous_{reason}",
        "stop_detail": details[reason],
    }


def executed_count_of(result: dict[str, Any]) -> int:
    if "executed_count" in result:
        return int(result.get("executed_count") or 0)
    return 1 if result.get("executed") else 0


def tool_output_limit(tool_output_tokens: int) -> int:
    return max(256, max(64, tool_output_tokens) * 4)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


class _Truncator:
    """Cuts fields to the limit and remembers whether any was cut (behaviour note F16)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.cut = False

    def __call__(self, text: str) -> str:
        if len(text) > self.limit:
            self.cut = True
        return _truncate(text, self.limit)

    def finish(self, body: dict[str, Any], tokens: int) -> dict[str, Any]:
        if self.cut:
            body["truncated"] = True
            body["truncation_note"] = (
                f"Part of this output was cut off so the reply stays within the roughly {tokens}-token "
                "budget for tool output."
            )
        return body


def _collapse_planned(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (len(item) if key == "planned_actions" and isinstance(item, list) else _collapse_planned(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_collapse_planned(item) for item in value]
    return value


def _elide_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "image_url", "image_url": {"url": f"<image {len(url)} chars>"}})
                else:
                    parts.append(part)
            message = {**message, "content": parts}
        out.append(message)
    return out


def _default_sandbox() -> tuple[Callable[..., dict[str, Any]], type[Exception]]:
    from wmagent.sandbox import ActionRefused, run_sandboxed_python

    return run_sandboxed_python, ActionRefused


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------


class ToolAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        gate: WorldModelGate | None | str = "env",
        sandbox: tuple[Callable[..., dict[str, Any]], type[Exception]] | None = None,
        number_of_levels: int | None = None,
        prompts_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CONFIG
        if not self.config.base_url:
            raise ValueError("No analyzer base URL: the base-URL variable in use is set but empty.")
        self.gate = WorldModelGate.from_env() if gate == "env" else gate
        self._sandbox = sandbox
        self.number_of_levels = number_of_levels
        self.prompts_dir = prompts_dir
        self.clock = clock
        self.http = requests.Session()
        self.system_prompt = prompts.system_prompt(self.gate_mode)
        self._state_path: Path | None = None
        self._new_game()

    @property
    def gate_mode(self) -> str | None:
        if self.gate is None:
            return None
        return "optional" if self.gate.optional else "forced"

    def _new_game(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.generated_tokens = 0
        self.total_tokens = 0
        self.uncached_input_tokens = 0
        self.last_summary: dict[str, Any] | None = None
        self.last_action_result: dict[str, Any] = {}
        self.notes: dict[str, str] = {}
        self.max_level = 1
        self.turns = 0
        self._recovered_ids = 0
        self._saved_history: list[dict[str, Any]] = []
        self._system_logged = False

    def sandbox(self) -> tuple[Callable[..., dict[str, Any]], type[Exception]]:
        if self._sandbox is None:
            self._sandbox = _default_sandbox()
        return self._sandbox

    # --- requests ----------------------------------------------------------

    def request_body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self.config
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "tools": TOOLS,
            "tool_choice": TOOL_CHOICE,
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
        }
        if cfg.max_output > 0:
            body["max_tokens"] = cfg.max_output
        if cfg.seed >= 0:
            body["seed"] = cfg.seed
        return body

    def send(self, messages: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        try:
            response = self.http.post(url, json=self.request_body(messages), timeout=max(0.1, timeout))
        except requests.RequestException as exc:
            raise AnalyzerRequestError(f"request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            message = f"HTTP {response.status_code} from {url}: {response.text}"
            raise (ContextLengthError if is_context_error(message) else AnalyzerRequestError)(message)
        try:
            data = response.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AnalyzerRequestError(f"unreadable response: {response.text[:500]}") from exc
        return {"message": choice.get("message") or {}, "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage") or {}}

    def _account(self, usage: dict[str, Any]) -> int:
        generated = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("generated_tokens") or 0
        prompt = usage.get("prompt_tokens") or 0
        total = usage.get("total_tokens") or (prompt + generated)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self.generated_tokens += int(generated)
        self.total_tokens += int(total)
        self.uncached_input_tokens += max(0, int(prompt) - int(cached))
        return int(generated)

    def _request_with_trim(self, messages: list[dict[str, Any]], timeout: float) -> tuple[dict[str, Any], list]:
        while True:
            try:
                return self.send(messages, timeout), messages
            except ContextLengthError:
                trimmed = trim_messages(messages, self.config.budget - SAFETY_TOKENS)
                if len(trimmed) == len(messages):
                    trimmed = [messages[0], *_user_first(drop_oldest_block(messages[1:]))]
                if len(trimmed) == len(messages):
                    raise
                messages = trimmed

    # --- the turn ----------------------------------------------------------

    def analyze(
        self,
        state_path: Path | str,
        action_count: int,
        valid_actions: list[str],
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        transcript_path: Path | str | None,
        analysis_step: int,
        request_timeout: float | Callable[[], float] = 900.0,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> dict[str, Any]:
        state_path = Path(state_path)
        transcript = Path(transcript_path) if transcript_path else None
        if state_path != self._state_path:
            self._state_path = state_path
            self._new_game()
            if self.gate is not None:
                self.gate.reset(state_path)
        saved_history = list(self.history)
        self._saved_history = saved_history
        try:
            return self._turn(state_path, action_count, [to_label(a) for a in valid_actions if str(a).upper() != "RESET"],
                              step_env, transcript, analysis_step, request_timeout, should_stop)
        except AnalyzerRequestError as exc:
            self.history = saved_history
            self._log(transcript, "STATUS", f"request error (history reverted): {exc}")
            return self._outcome(False, retryable=True, reasoning="", error=str(exc))
        except BaseException:
            self.history = saved_history
            raise

    @staticmethod
    def _outcome(executed: bool, *, retryable: bool = False, yielded: bool = False, reasoning: str = "",
                 error: str | None = None, yield_reason: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"step_executed": executed, "retryable_failure": retryable,
                               "yielded_control": yielded, "reasoning": reasoning}
        if error is not None:
            out["error"] = error
        if yield_reason is not None:
            out["yield_reason"] = yield_reason
        return out

    def _turn(self, state_path, action_count, valid_actions, step_env, transcript, step, request_timeout, should_stop):
        started = self.clock()
        self.turns += 1
        runtime = load_runtime_state(state_path)
        frame = runtime.get("current_frame") or {}
        self.max_level = max(self.max_level, int(frame.get("level") or 1))
        text = prompts.user_prompt(
            step=step, level=self.max_level, number_of_levels=self.number_of_levels, action_count=action_count,
            valid_actions=valid_actions, notes=self.notes, previous_summary=self.last_summary,
            first_turn=not self.history and self.last_summary is None,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if self.config.multimodal_context == "current_grid" and frame.get("grid"):
            url = grid_png_data_url(frame["grid"], self.config.multimodal_upscale)
            content.append({"type": "image_url", "image_url": {"url": url}})
        if not self._system_logged:
            self._log(transcript, "SYSTEM PROMPT", self.system_prompt)
            self._system_logged = True
        self._log(transcript, "USER PROMPT", text)

        system = {"role": "system", "content": self.system_prompt}
        messages = trim_messages([system, *self.history, {"role": "user", "content": content}], self.config.budget)
        ctx = _TurnContext(valid_actions=valid_actions, step_env=step_env, state_path=state_path)
        reasoning_seen: list[str] = []

        def yield_reason() -> str | None:
            if should_stop():
                return "stop_requested"
            # LOCAL_ANALYZER_YIELD_SECONDS <= 0 disables the turn budget, as in the original
            if self.config.yield_seconds > 0 and self.clock() - started >= self.config.yield_seconds:
                return "turn_time_budget"
            return None

        requests_made = 0
        while True:
            reason = yield_reason()
            if reason:
                return self._finish_turn(messages, ctx, reasoning_seen, transcript, yielded=reason)
            if self.config.tool_steps > 0 and requests_made >= self.config.tool_steps:
                return self._finish_turn(messages, ctx, reasoning_seen, transcript, yielded="tool_step_limit")
            timeout = request_timeout() if callable(request_timeout) else request_timeout
            messages = trim_messages(messages, self.config.budget)
            self._snapshot(state_path, step, requests_made, messages)
            reply, messages = self._request_with_trim(messages, timeout)
            requests_made += 1
            generated = self._account(reply["usage"])
            message = reply["message"]
            raw_content = message.get("content") or ""
            raw_reasoning = clean_text(message.get("reasoning") or message.get("reasoning_content") or "")
            # Tool-call markup is looked for in both the reasoning and the content (as in the original),
            # recovered from both (deduplicated) and stripped from both.
            markup_in_reasoning = bool(MARKUP_DETECT.search(raw_reasoning))
            markup_in_content = bool(MARKUP_DETECT.search(raw_content))
            markup_seen = markup_in_reasoning or markup_in_content
            calls = [
                {"id": call.get("id") or self._next_call_id(), "name": (call.get("function") or {}).get("name", ""),
                 "arguments": _parse_arguments((call.get("function") or {}).get("arguments")),
                 "raw_arguments": (call.get("function") or {}).get("arguments")}
                for call in message.get("tool_calls") or []
            ]
            if not calls and markup_seen:
                calls = [
                    {"id": f"markup-call-{index}", **call}
                    for index, call in enumerate(_dedup_calls(
                        recover_tool_calls(raw_reasoning) + recover_tool_calls(raw_content)), start=1)
                ]
            visible = strip_markup(raw_content) if markup_in_content else clean_text(raw_content)
            reasoning = strip_markup(raw_reasoning) if markup_in_reasoning else raw_reasoning
            self._log(transcript, "RESPONSE META", json.dumps(
                {"finish_reason": reply["finish_reason"], "generated_tokens": generated, "tool_calls": len(calls)}))
            if reasoning:
                reasoning_seen.append(reasoning)
                self._log(transcript, "THINKING", reasoning)
            if visible:
                self._log(transcript, "ASSISTANT", visible)
                self.notes.update(extract_notes(visible))

            if not calls:
                if visible or reasoning:
                    messages.append(self._assistant_message(visible, reasoning, []))
                reason = yield_reason()
                if reason:
                    return self._finish_turn(messages, ctx, reasoning_seen, transcript, yielded=reason)
                corrective = prompts.corrective_prompt(markup_seen)
                self._log(transcript, "USER PROMPT", corrective)
                messages.append({"role": "user", "content": corrective})
                continue

            messages.append(self._assistant_message(visible, reasoning, calls))
            for index, call in enumerate(calls):
                self._log(transcript, "TOOL CALL", f"{call['name']}: {json.dumps(call['arguments'])}")
                output = self._run_tool(call, ctx)
                self._log(transcript, "TOOL RESULT", output)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                # If later calls of this response get skipped, the turn's history is not persisted
                # (the original reverts it), so no assistant message with unanswered tool_calls is kept.
                skipped = index < len(calls) - 1
                if ctx.executed_any:
                    return self._finish_turn(messages, ctx, reasoning_seen, transcript, persist=not skipped)
                reason = yield_reason()
                if reason:
                    return self._finish_turn(messages, ctx, reasoning_seen, transcript, yielded=reason,
                                             persist=not skipped)

    def _next_call_id(self) -> str:
        self._recovered_ids += 1
        return f"call_{self._recovered_ids}"

    @staticmethod
    def _assistant_message(content: str, reasoning: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
        # content only when non-empty, and the server's own argument strings are re-sent unchanged
        message: dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = content
        if reasoning:
            message["reasoning"] = reasoning
        if calls:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["raw_arguments"]
                              if isinstance(c.get("raw_arguments"), str) else json.dumps(c["arguments"])}}
                for c in calls
            ]
        return message

    def _finish_turn(self, messages, ctx: _TurnContext, reasoning_seen, transcript, yielded: str | None = None,
                     persist: bool = True):
        if persist:
            self.history = persistable_history(messages, self.config.budget)
        else:
            self.history = list(self._saved_history)
        if ctx.results:
            self.last_summary = summarize(ctx.results, ctx.start_level)
            self.last_action_result = ctx.results[-1]
            self.max_level = max(self.max_level, int(self.last_summary.get("level") or 1))
            if any(self.last_summary.get(flag) for flag in ("level_transition", "run_complete", "game_over")):
                for key in prompts.LEVEL_SCOPED_NOTES:
                    self.notes.pop(key, None)
        reasoning = "\n".join(reasoning_seen)
        if yielded and not ctx.executed_any:
            self._log(transcript, "STATUS", f"yield: {yielded}")
            return self._outcome(False, yielded=True, reasoning=reasoning, yield_reason=yielded)
        self._log(transcript, "STATUS", f"turn done: executed={ctx.executed_any}")
        return self._outcome(ctx.executed_any, reasoning=reasoning)

    # --- python tool -------------------------------------------------------

    def _run_tool(self, call: dict[str, Any], ctx: _TurnContext) -> str:
        limit = self.config.tool_output_chars
        if call["name"] != "python":
            return json.dumps({"tool": call["name"], "error": f"Unknown tool '{call['name']}'. Use python."}, indent=2)
        code = call["arguments"].get("code")
        if not isinstance(code, str) or not code.strip():
            return json.dumps({"error": "Empty code: pass Python source in `code`."}, indent=2)
        code = code.rstrip()  # the original rstrips the cell before running it and before after_cell
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return json.dumps({"error": f"Python syntax error: {exc}"}, indent=2)

        run_python, refused = self.sandbox()
        cell = _Cell()

        def handle(actions: Any) -> dict[str, Any]:
            items = actions if isinstance(actions, list) else [actions]
            items = [{"action": a} if isinstance(a, str) else a for a in items]
            if cell.latched:
                # no step_env and no gate; the stub is not compacted
                stub = latch_stub(cell.latched_result, cell.latched, len(items))
                cell.results.append(stub)
                ctx.valid_actions = []
                self.last_action_result = stub
                return {"action_result": stub, "state": self._state(ctx)}
            # The gate's action accounting lives here, as in the original: the level is observed
            # only before the check, so a level completed by this batch is still scored on the old
            # level by after_cell. A refusal raises out of the handler and the sandbox surfaces it
            # to model code as RuntimeError(msg).
            if self.gate is not None:
                self.gate.observe_level(_current_level(ctx.state_path))
                refusal = self.gate.check_action(len(items))
                if refusal is not None:
                    raise refused(refusal)
            payload = ctx.step_env({"actions": items})
            result = compact_result(payload)
            if self.gate is not None:
                # a failed call (error payload, no executed_count) counts 1, as in the original
                self.gate.record_actions(int(result.get("executed_count", 1) or 0))
            executed = executed_count_of(payload)
            cell.results.append(result)
            if executed > 0:
                ctx.results.append(result)
                ctx.executed_any = True
            for flag in TERMINAL_FLAGS if result.get("executed") else ():
                if result.get(flag):
                    cell.latched = flag
                    cell.latched_result = result
                    break
            if isinstance(payload.get("valid_actions"), list):
                ctx.valid_actions = list(payload["valid_actions"])
            self.last_action_result = result
            return {"action_result": result, "state": self._state(ctx)}

        to_run = self.gate.prepend(code) if self.gate is not None else code
        outcome = run_python(code=to_run, timeout_seconds=self.config.tool_timeout,
                             initial_state=self._state(ctx), action_handler=handle)
        had_error = bool(outcome.get("error"))
        if self.gate is not None:
            self.gate.after_cell(code, had_error, lambda src: self._backtest(src, ctx))
        return format_tool_result(outcome, cell.results, limit, self.config.tool_output_tokens)

    def _state(self, ctx: _TurnContext) -> dict[str, Any]:
        return sandbox_state(load_runtime_state(ctx.state_path), ctx.valid_actions, self.last_action_result)

    def _backtest(self, source: str, ctx: _TurnContext) -> list[list[Any]] | None:
        run_python, refused_cls = self.sandbox()

        def refuse(_actions: Any) -> dict[str, Any]:
            raise refused_cls("Backtest cells cannot take actions.")

        outcome = run_python(code=source + "\n\n" + BACKTEST_DRIVER, timeout_seconds=self.config.tool_timeout,
                             initial_state=self._state(ctx), action_handler=refuse)
        if outcome.get("error"):
            self.gate.note_backtest_error(str(outcome["error"]))
            return None
        return outcome.get("result")

    # --- logging -----------------------------------------------------------

    def _log(self, transcript: Path | None, section: str, body: str) -> None:
        if transcript is None:
            return
        transcript.parent.mkdir(parents=True, exist_ok=True)
        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {section} =====\n{body}\n")

    def _snapshot(self, state_path: Path, step: int, request_index: int, messages: list[dict[str, Any]]) -> None:
        if self.prompts_dir is None:
            return
        stem = state_path.name.removesuffix(".json").removesuffix("_tool_runtime_state")
        folder = Path(self.prompts_dir) / stem
        folder.mkdir(parents=True, exist_ok=True)
        record = {"step": step, "request": request_index, "message_count": len(messages),
                  "estimated_tokens": round(estimate_tokens(messages)), "messages": _elide_images(messages)}
        (folder / f"step_{step:04d}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


@dataclass
class _Cell:
    latched: str | None = None
    latched_result: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)


class _TurnContext:
    def __init__(self, valid_actions: list[str], step_env: Callable, state_path: Path) -> None:
        self.valid_actions = list(valid_actions)
        self.step_env = step_env
        self.state_path = state_path
        self.results: list[dict[str, Any]] = []
        self.executed_any = False
        runtime = load_runtime_state(state_path)
        self.start_level = int((runtime.get("current_frame") or {}).get("level") or 1)


def summarize(results: list[dict[str, Any]], start_level: int) -> dict[str, Any]:
    labels: list[str] = []
    for result in results:
        executed = result.get("executed_actions")
        if isinstance(executed, list) and executed:
            labels.extend(str(a) for a in executed)
        elif result.get("action_display"):
            labels.append(str(result["action_display"]))
    count = sum(executed_count_of(r) for r in results)
    first, last = results[0], results[-1]
    end_num = last.get("action_num")
    start_num = None
    if isinstance(first.get("action_num"), int):
        start_num = first["action_num"] - executed_count_of(first) + 1
    return {
        "executed_count": count,
        "executed_actions": labels,
        "start_action_num": start_num,
        "end_action_num": end_num,
        "level": last.get("level"),
        "state": last.get("state"),
        "board_changed": any(r.get("board_changed") for r in results),
        "level_transition": any(r.get("level_completed") for r in results)
        or (last.get("level") is not None and last.get("level") != start_level),
        "run_complete": any(r.get("run_complete") for r in results),
        "game_over": any(r.get("game_over") for r in results),
        "stop_reason": last.get("stop_reason"),
    }


def format_tool_result(outcome: dict[str, Any], action_results: list[dict[str, Any]], limit: int,
                       tool_output_tokens: int | None = None) -> str:
    tokens = tool_output_tokens if tool_output_tokens is not None else limit // 4
    cut = _Truncator(limit)
    stdout = outcome.get("stdout") or ""
    if outcome.get("error"):
        body: dict[str, Any] = {"tool": "python", "error": cut(str(outcome["error"]))}
        if stdout:
            body["stdout"] = cut(str(stdout))
        return json.dumps(cut.finish(body, tokens), indent=2)
    # Exactly one of: stdout, else result, else the single action result, else the action summary;
    # the last two go under `result` too (the original's precedence).
    body = {"tool": "python", "returncode": 0}
    result = _collapse_planned(outcome.get("result"))
    if stdout:
        body["stdout"] = cut(str(stdout))
    elif result is not None:
        body["result"] = cut(result) if isinstance(result, str) else result
    elif len(action_results) == 1:
        body["result"] = _collapse_planned(action_results[0])
    elif action_results:
        body["result"] = {"action_calls": len(action_results),
                          "last_action_result": _collapse_planned(action_results[-1])}
    return json.dumps(cut.finish(body, tokens), indent=2, default=str)
