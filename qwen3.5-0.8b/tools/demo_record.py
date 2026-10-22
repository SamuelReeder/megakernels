"""Record timestamped token streams from the megakernel and llama.cpp for the README demo.

Both engines get the same chat prompt through their own streaming path, and every chunk is stamped
when it reaches the host, so the recording holds real wall-clock timing. `demo_render.py` turns it
into a side-by-side GIF. Run it with the GPU otherwise idle (see tools/wait_gpu_quiet.sh).

    python tools/demo_record.py --llama-server ~/src/llama.cpp-latest/build/bin/llama-server \
        --gguf ~/models/qwen35/Qwen3.5-0.8B-Q8_0.gguf --out results/demo_recording.json
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def record_megakernel(n_tokens, seed):
    from qwen_mk.session import ChatSession

    s = ChatSession(quant="int8")
    msgs = [{"role": "user", "content": PROMPT}]
    for _ in s.chat([{"role": "user", "content": "Hi"}], max_new_tokens=16, temperature=0.7, seed=1):
        pass  # warm up: kernel compile and caches; a different prompt, so no state is reused below
    events, ids = [], []
    t0 = time.perf_counter()
    for t in s.chat(msgs, max_new_tokens=n_tokens, temperature=0.7, seed=seed):
        ids.append(t)
        events.append((time.perf_counter() - t0, s.tok.decode(ids)))
    return dict(label="megakernel int8", events=events, n_tokens=len(ids))


def record_llama(server, gguf, n_tokens, seed, port=8932):
    proc = subprocess.Popen([server, "-m", gguf, "-ngl", "99", "-fa", "on", "-c", "4096", "-np", "1", "--port",
                             str(port), "--host", "127.0.0.1", "--no-webui"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except Exception:
                time.sleep(1)

        def go(max_tokens, prompt=PROMPT):
            body = json.dumps(dict(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                                   temperature=0.7, seed=seed, stream=True, cache_prompt=False,
                                   chat_template_kwargs={"enable_thinking": False})).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            events, text = [], ""
            t0 = time.perf_counter()
            with urllib.request.urlopen(req) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    ch = json.loads(line[5:])["choices"]
                    piece = ch[0]["delta"].get("content") if ch else None
                    if piece:
                        text += piece
                        events.append((time.perf_counter() - t0, text))
            return events

        go(16, "Hi")  # warm up
        events = go(n_tokens)
        return dict(label="llama.cpp Q8_0", events=events, n_tokens=len(events))
    finally:
        proc.terminate()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-server", default=os.path.expanduser("~/src/llama.cpp-latest/build/bin/llama-server"))
    ap.add_argument("--gguf", default=os.path.expanduser("~/models/qwen35/Qwen3.5-0.8B-Q8_0.gguf"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/demo_recording.json")
    args = ap.parse_args()
    runs = [record_megakernel(args.tokens, args.seed), record_llama(args.llama_server, args.gguf, args.tokens, args.seed)]
    for r in runs:
        dur = r["events"][-1][0] - r["events"][0][0]
        print(f"{r['label']}: {r['n_tokens']} tokens, {r['n_tokens'] / dur:.0f} tok/s after the first token")
    json.dump(dict(prompt=PROMPT, runs=runs), open(args.out, "w"))


if __name__ == "__main__":
    main()
