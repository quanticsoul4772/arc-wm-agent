"""Scripted OpenAI-compatible chat server for agent tests and record/replay (D2).

Each POST to /v1/chat/completions pops the next scripted entry, ignoring the
request content (which is recorded in `.requests`). An entry is either a full
response body ({"choices": [...], "usage": {...}}), a shorthand built with
`tool_call_response` / `text_response`, or an error {"status": int, "body": ...}.
When the script is exhausted the server answers 503 and sets `.exhausted`.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self


def _usage(completion: int = 10, prompt: int = 100) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def chat_body(message: dict[str, Any], finish_reason: str = "stop", usage: dict | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
        "usage": usage or _usage(),
    }


def tool_call_response(code: str, call_id: str = "call_a", content: str = "", reasoning: str | None = None,
                       completion_tokens: int = 10) -> dict[str, Any]:
    message: dict[str, Any] = {
        "content": content,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "python", "arguments": json.dumps({"code": code})}}],
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return chat_body(message, "tool_calls", _usage(completion_tokens))


def text_response(content: str, reasoning: str | None = None, completion_tokens: int = 10) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return chat_body(message, "stop", _usage(completion_tokens))


def error_response(status: int, body: Any) -> dict[str, Any]:
    return {"status": status, "body": body}


class FakeOpenAIServer:
    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self.requests: list[dict[str, Any]] = []
        self.exhausted = False
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:  # /v1/models for readiness polling
                self._send(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {"_raw": raw.decode("utf-8", "replace")}
                status, payload = server._next(body)
                self._send(status, payload)

            def _send(self, status: int, payload: Any) -> None:
                data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> FakeOpenAIServer:
        """Replay recorded responses, one JSON object per line, in order."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([json.loads(line) for line in lines if line.strip()])

    def _next(self, body: dict[str, Any]) -> tuple[int, Any]:
        with self._lock:
            self.requests.append(body)
            if not self.script:
                self.exhausted = True
                return 503, {"error": {"message": "fake server script exhausted"}}
            entry = self.script.pop(0)
        if "status" in entry and "choices" not in entry:
            return int(entry["status"]), entry.get("body", "")
        return 200, entry

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
