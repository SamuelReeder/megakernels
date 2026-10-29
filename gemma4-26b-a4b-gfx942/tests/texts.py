"""Real-text passages for the reference and kernel checks (public domain or written here)."""

TEXTS = [
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
    "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the "
    "season of Light, it was the season of Darkness, it was the spring of hope, it was the winter "
    "of despair, we had everything before us, we had nothing before us, we were all going direct to "
    "Heaven, we were all going direct the other way - in short, the period was so far like the "
    "present period, that some of its noisiest authorities insisted on its being received, for good "
    "or for evil, in the superlative degree of comparison only.",
    "A hash table stores key-value pairs in an array of buckets. A hash function maps each key to a "
    "bucket index. When two keys map to the same bucket, a collision occurs; separate chaining keeps "
    "a linked list per bucket, while open addressing probes other buckets until it finds a free one. "
    "With a good hash function and a load factor below one, lookups, insertions and deletions take "
    "constant time on average. Resizing the table when the load factor grows keeps that guarantee.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in "
    "range(n):\n        a, b = b, a + b\n    return a\n\n\nif __name__ == \"__main__\":\n    for i in "
    "range(10):\n        print(i, fibonacci(i))\n",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must "
    "be in want of a wife. However little known the feelings or views of such a man may be on his first "
    "entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that "
    "he is considered the rightful property of some one or other of their daughters. \"My dear Mr. "
    "Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. "
    "Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and "
    "she told me all about it.\" Mr. Bennet made no answer. \"Do you not want to know who has taken it?\" "
    "cried his wife impatiently. \"You want to tell me, and I have no objection to hearing it.\" This was "
    "invitation enough.",
    "The mitochondrion is the site of aerobic respiration in eukaryotic cells. Pyruvate from "
    "glycolysis enters the matrix, where the citric acid cycle oxidizes it to carbon dioxide and "
    "transfers electrons to NADH and FADH2. The electron transport chain in the inner membrane then "
    "pumps protons across the membrane, and ATP synthase uses the resulting gradient to make ATP.",
]


def long_ids(tok, n: int) -> list[int]:
    """At least n token ids (BOS first): the passages in turn, numbered, so no two windows
    of the text repeat exactly. Truncated to n."""
    ids = [tok.bos_token_id]
    r = 0
    while len(ids) < n:
        for i, t in enumerate(TEXTS):
            ids += tok(f"\n\n[{r}.{i}] {t}", add_special_tokens=False).input_ids
        r += 1
    return ids[:n]


def text_prompts(model_dir: str, M: int, ctx: int):
    """[ctx][M] int64 on cuda: row r is the window of long_ids starting at 37 r (the rows
    of bench/vllm_e2e.py's speed test use the same offsets; distinct real text routes
    to more distinct experts than random tokens do)."""
    import torch
    from transformers import AutoTokenizer

    ids = long_ids(AutoTokenizer.from_pretrained(model_dir), ctx + 37 * M)
    rows = [ids[37 * r: 37 * r + ctx] for r in range(M)]
    return torch.tensor(rows, device="cuda").t().contiguous()
