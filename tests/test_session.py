"""Session-level behaviour: HF prefill state transfer and snapshot reuse across turns."""

import pytest

from qwen_mk import session as session_mod
from qwen_mk.session import ChatSession

LONG = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(60))


@pytest.fixture(scope="module")
def s():
    return ChatSession(max_seq=4096)


def greedy(s, messages, n=24):
    return list(s.chat(messages, max_new_tokens=n, temperature=0.0))


def test_hf_prefill_matches_megakernel_prefill(s):
    msgs = [{"role": "system", "content": LONG}, {"role": "user", "content": "Summarize item 3 in one line."}]
    ids = s.encode(s.render(msgs))
    assert len(ids) > session_mod.HF_PREFILL_MIN
    s.hf_prefill, s.fed, s.snapshots = False, None, []
    want = greedy(s, msgs)
    s.hf_prefill, s.fed, s.snapshots = True, None, []
    got = greedy(s, msgs)
    match = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want))
    assert match >= 16, (s.tok.decode(got), s.tok.decode(want))


def test_multi_turn_reuses_snapshot(s):
    s.fed, s.snapshots = None, []
    msgs = [{"role": "user", "content": "Name three primary colors."}]
    reply = s.tok.decode(greedy(s, msgs, 40), skip_special_tokens=True)
    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Now name one secondary color."}]
    second = greedy(s, msgs, 24)
    # Only the re-rendered last assistant turn + new user turn should be fed, not the whole chat.
    assert s.stats["prefill_tokens"] < len(s.encode(s.render(msgs))) - 5
    s.fed, s.snapshots = None, []
    fresh = greedy(s, msgs, 24)
    assert second == fresh
