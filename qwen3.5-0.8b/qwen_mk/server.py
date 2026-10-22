"""OpenAI-compatible HTTP server for Qwen3.5-0.8B on the megakernel (stdlib only).

    qwen-mk-serve --port 8000
    curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'

Supports /v1/models and /v1/chat/completions (streaming and not, tools via Qwen's XML
tool-call format, `enable_thinking`). One GPU, batch size 1: requests are served one at a
time, and consecutive requests from the same conversation reuse the recurrent state.
"""

import argparse
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import TextStream
from .session import ChatSession

MODEL_ID = "qwen3.5-0.8b-megakernel"
TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def normalize_messages(messages):
    """OpenAI message shapes -> what the Qwen chat template expects."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if p.get("type") in (None, "text"))
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", c))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        fn["arguments"] = {"input": fn["arguments"]}
                calls.append({"type": "function", "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def parse_tool_calls(text):
    calls = []
    for name, body in TOOL_RE.findall(text):
        args = {}
        for key, val in PARAM_RE.findall(body):
            try:
                args[key] = json.loads(val)
            except json.JSONDecodeError:
                args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    content = TOOL_RE.sub("", text).strip() if calls else text
    return content, calls


def split_reasoning(text):
    if "</think>" in text:
        reasoning, _, rest = text.partition("</think>")
        return rest.strip(), reasoning.replace("<think>", "").strip()
    return text, None


class Handler(BaseHTTPRequestHandler):
    session: ChatSession = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[qwen-mk] " + fmt % args + "\n")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        elif self.path.rstrip("/") in ("", "/health"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            messages = normalize_messages(req["messages"])
        except (ValueError, KeyError) as e:
            return self._json(400, {"error": {"message": f"bad request: {e}"}})
        s = self.session
        tools = req.get("tools") or None
        think = bool(req.get("enable_thinking", (req.get("chat_template_kwargs") or {}).get("enable_thinking", False)))
        prompt = s.encode(s.render(messages, enable_thinking=think, tools=tools))
        max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 2048)
        temperature = float(req.get("temperature", 0.7))
        seed = req.get("seed")
        cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        gen = s.stream(prompt, max_new_tokens=max_new, temperature=temperature, seed=seed)

        if not req.get("stream"):
            ids = list(gen)
            finish = "stop" if ids and ids[-1] in s.stop_ids else "length"
            text = s.tok.decode([t for t in ids if t not in s.stop_ids], skip_special_tokens=False)
            text, reasoning = split_reasoning(text)
            content, calls = parse_tool_calls(text)
            msg = {"role": "assistant", "content": content or (None if calls else "")}
            if calls:
                msg["tool_calls"] = calls
                finish = "tool_calls"
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids),
                          "total_tokens": len(prompt) + len(ids)},
            })

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(delta, finish=None):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            data = f"data: {json.dumps(chunk)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            send({"role": "assistant", "content": ""})
            out, text = TextStream(s.tok), ""
            sent_r = sent_c = 0         # characters of reasoning / content already sent
            body_start = None if think else 0
            finish = "length"
            for t in gen:  # a stop id is always the last token; don't break, so `gen` records its state
                if t in s.stop_ids:
                    finish = "stop"
                    continue
                text += out.push(t)
                if body_start is None:  # still inside the thinking block
                    reasoning = text.split("</think>")[0].replace("<think>", "")
                    if reasoning[sent_r:]:
                        send({"reasoning_content": reasoning[sent_r:]})
                        sent_r = len(reasoning)
                    if "</think>" in text:
                        body_start = text.index("</think>") + len("</think>")
                    continue
                body = text[body_start:].lstrip("\n") if sent_c == 0 else text[body_start:]
                if sent_c == 0:
                    body_start = len(text) - len(body)
                visible = body.split("<tool_call>")[0] if tools else body
                if visible[sent_c:]:
                    send({"content": visible[sent_c:]})
                    sent_c = len(visible)
            if body_start is not None and tools:
                content, calls = parse_tool_calls(text[body_start:])
                if calls:
                    send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                    finish = "tool_calls"
                elif content[sent_c:]:
                    send({"content": content[sent_c:]})
            send({}, finish)
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            gen.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--quant", choices=["int8", "bf16"], default="int8",
                    help="int8 (default): 1.6x faster, +0.1%% WikiText perplexity; bf16: exact weights")
    ap.add_argument("--no-spec", action="store_true",
                    help="decode one token per step instead of speculatively with the MTP head (same output)")
    args = ap.parse_args()
    print("loading model (the first run also compiles the kernel, ~1 min)...", file=sys.stderr)
    Handler.session = ChatSession(args.model, max_seq=args.max_seq, quant=args.quant, spec=not args.no_spec)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {MODEL_ID} on http://{args.host}:{args.port}/v1", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
