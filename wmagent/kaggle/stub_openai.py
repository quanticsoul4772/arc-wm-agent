"""A stub OpenAI-compatible server for local notebook runs without a GPU (stdlib only).

Serves GET /v1/models and POST /v1/chat/completions. Every chat reply carries
reasoning, a world-model note and one `python` tool call that takes the first
valid non-MOUSE action (MOUSE at the center when that is the only option), so an
agent driven by it plays legal actions. The smoke chat (no tools) gets "OK".

    python -m wmagent.kaggle.stub_openai --host 127.0.0.1 --port 1234
"""
from __future__ import annotations

import argparse
import itertools
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "vrfai/Qwen3.6-27B-FP8"
ACTION_CODE = (
    "acts = [a for a in valid_actions if a != 'MOUSE']\n"
    "r = action(acts[0] if acts else {'action': 'MOUSE', 'row': 32, 'col': 32})\n"
    "print(r)\n"
)
_ids = itertools.count(1)


def chat_reply(body: dict) -> dict:
    n = next(_ids)
    if body.get("tools"):
        message = {
            "role": "assistant",
            "reasoning": "Stub reasoning: take the first valid action.",
            "content": "World model: unknown (stub).\nPlan: take the first valid action.",
            "tool_calls": [{
                "id": f"call_{n}", "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": ACTION_CODE})},
            }],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": "OK"}
        finish = "stop"
    return {
        "id": f"chatcmpl-stub-{n}", "object": "chat.completion", "model": body.get("model", MODEL),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, data: dict) -> None:
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self._send(200, chat_reply(body))

    def log_message(self, fmt: str, *args) -> None:
        print("[stub_openai] " + fmt % args, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1234)
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[stub_openai] serving on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
