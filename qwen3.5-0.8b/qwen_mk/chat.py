"""Interactive chat with Qwen3.5-0.8B running on the megakernel.

    qwen-mk-chat [--temperature 0.7] [--system "You are ..."] [--think] [--quant int8|bf16]

Commands: /reset clears the conversation, /stats toggles speed stats, /quit exits.
"""

import argparse
import sys

from .session import ChatSession


class TextStream:
    """Incremental detokenizer: yields text as soon as it no longer ends mid-character."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, t):
        self.ids.append(t)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):
            return ""
        new, self.sent = text[self.sent:], len(text)
        return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--system", default=None)
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--quant", choices=["int8", "bf16"], default="int8",
                    help="int8 (default): 1.6x faster, +0.1%% WikiText perplexity; bf16: exact weights")
    ap.add_argument("--no-spec", action="store_true",
                    help="decode one token per step instead of speculatively with the MTP head (same output)")
    args = ap.parse_args()

    print("loading model (the first run also compiles the kernel, ~1 min)...", file=sys.stderr)
    s = ChatSession(args.model, max_seq=args.max_seq, quant=args.quant, spec=not args.no_spec)
    show_stats = True
    messages = [{"role": "system", "content": args.system}] if args.system else []
    print("ready. /reset, /stats, /quit", file=sys.stderr)
    while True:
        try:
            user = input("\n\033[1m> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        cmd = user.strip()
        if cmd in ("/quit", "/exit"):
            return
        if cmd == "/reset":
            messages = messages[:1] if args.system else []
            print("(conversation cleared)")
            continue
        if cmd == "/stats":
            show_stats = not show_stats
            continue
        if not cmd:
            continue
        messages.append({"role": "user", "content": user})
        out = TextStream(s.tok)
        reply = []
        try:
            for t in s.chat(messages, enable_thinking=args.think, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature):
                reply.append(t)
                sys.stdout.write(out.push(t))
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        text = s.tok.decode(reply, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text.split("</think>")[-1].strip()})
        if show_stats and s.stats:
            st = s.stats
            print(f"\n\033[2m[{st['prefill_tokens']} prompt tokens in {st['prefill_s'] * 1e3:.0f} ms | "
                  f"{st['decode_tokens']} tokens at {st['decode_tok_s']:.0f} tok/s]\033[0m")


if __name__ == "__main__":
    main()
