"""Render results/demo_recording.json (from demo_record.py) as a side-by-side GIF.

The recording is replayed with its real timestamps, slowed down so the race is watchable, and the
tok/s counters are computed from real (not slowed) time.

    python tools/demo_render.py --out docs/demo.gif
"""

import argparse
import json
import os
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

BG, PANE, INK, MUTED, GRID = "#14151a", "#1b1d24", "#e6e6e1", "#8b8d96", "#2a2c35"
ACCENT = {"megakernel int8": "#4d9bf0", "llama.cpp Q8_0": "#f0835a"}
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
W, H, FPS = 1280, 640, 20


def text_at(events, t):
    cur, n = "", 0
    for i, (ts, text) in enumerate(events):
        if ts > t:
            break
        cur, n = text, i + 1
    return cur, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="results/demo_recording.json")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--slow", type=float, default=0.2, help="playback speed relative to real time")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to hold the final frame")
    args = ap.parse_args()
    rec = json.load(open(args.rec))
    runs = rec["runs"]
    f, fb, fs = (ImageFont.truetype(MONO, 15), ImageFont.truetype(MONO_B, 17), ImageFont.truetype(MONO, 13))
    cw = f.getlength("M")
    pane_w = (W - 3 * 20) // 2
    cols = int((pane_w - 32) / cw)
    rows = 15
    end = max(r["events"][-1][0] for r in runs)
    frames = int(end / args.slow * FPS) + int(args.hold * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(frames):
            t = min(k / FPS * args.slow, end)
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            d.text((20, 16), "Qwen3.5-0.8B on one RTX 5080, same prompt, batch 1", font=fb, fill=INK)
            d.text((20, 42), f"> {rec['prompt']}", font=f, fill=MUTED)
            d.text((W - 20 - fs.getlength("replayed at 0.2x speed; counters use real time"), 18),
                   f"replayed at {args.slow:g}x speed; counters use real time", font=fs, fill=MUTED)
            for j, r in enumerate(runs):
                x0 = 20 + j * (pane_w + 20)
                y0 = 78
                col = ACCENT[r["label"]]
                d.rounded_rectangle((x0, y0, x0 + pane_w, H - 20), 8, fill=PANE)
                d.rectangle((x0, y0 + 8, x0 + 4, y0 + 34), fill=col)
                d.text((x0 + 16, y0 + 10), r["label"], font=fb, fill=col)
                text, n = text_at(r["events"], t)
                lines = []
                for para in text.split("\n"):
                    lines += textwrap.wrap(para, cols) or [""]
                for i, line in enumerate(lines[-rows:]):
                    d.text((x0 + 16, y0 + 52 + i * 21), line, font=f, fill=INK)
                first = r["events"][0][0]
                rate = n / (min(t, r["events"][-1][0]) - first) if n > 1 and t > first else 0.0
                done = n >= r["n_tokens"]
                d.line((x0 + 16, H - 76, x0 + pane_w - 16, H - 76), fill=GRID)
                d.text((x0 + 16, H - 62), f"{n:4d} tokens", font=f, fill=MUTED)
                big = f"{rate:4.0f} tok/s"
                d.text((x0 + pane_w - 16 - fb.getlength(big), H - 64), big, font=fb, fill=col if done else INK)
                if done:
                    d.text((x0 + 16, H - 40), f"finished in {r['events'][-1][0] * 1000:.0f} ms", font=fs, fill=col)
            im.save(os.path.join(tmp, f"f{k:04d}.png"))
        pal = os.path.join(tmp, "pal.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                        "-vf", "palettegen=max_colors=64", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png", "-i",
                        pal, "-lavfi", "paletteuse=dither=none", "-loop", "0", args.out], check=True)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {frames} frames)")


if __name__ == "__main__":
    main()
