"""bench/results/2026-10-07-conversation-slots/conv_switch.py - deterministic long chats against a running server, to time
conversation switches.  --plan A,B,A: one request each, in order; A|B: A runs and B queues behind it (a HINT)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone cloud forest lantern copper "
         "violet harbor meadow signal engine planet orbit falcon ember glacier summit canyon prairie willow cedar "
         "maple saffron indigo crimson amber cobalt garnet onyx quartz topaz jasper basalt granite marble").split()


def text(seed: int, words: int) -> str:
    r = random.Random(seed)
    lines = []
    for i in range(0, words, 12):
        lines.append(f"[{seed}:{i}] " + " ".join(r.choice(WORDS) for _ in range(12)))
    return "\n".join(lines)


def chat(url, messages, max_tokens=24):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    msg = out["choices"][0]["message"]["content"]
    return msg, out.get("timings") or {}, wall


class Conv:
    def __init__(self, name, seed, words, long=False):
        self.name, self.long = name, long
        self.messages = [{"role": "system", "content": "You are a terse assistant. Answer in one short line."},
                         {"role": "user", "content": f"Document {name}:\n{text(seed, words)}\n\nWhat is the first "
                                                     f"word after [{seed}:0]?"}]
        self.turn = 0

    def step(self, url, max_tokens=24):
        msg, tm, wall = chat(url, self.messages, max_tokens)
        self.messages.append({"role": "assistant", "content": msg})
        self.turn += 1
        ask = (f"Turn {self.turn}: write a 300-word story that uses at least twenty words from document {self.name}."
               if self.long else
               f"Turn {self.turn}: repeat your previous answer, then add one word from document {self.name}.")
        self.messages.append({"role": "user", "content": ask})
        return msg, tm, wall


def report(tag, name, msg, tm, wall):
    print(json.dumps({"tag": tag, "conv": name, "wall_s": round(wall, 2), "cache_n": tm.get("cache_n"),
                      "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                      "decode_tok_s": tm.get("predicted_per_second"), "text": msg}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--words", type=int, default=3000)
    ap.add_argument("--plan", required=True, help="e.g. A,B,A,B or A,A or A,B,C,A,B,C; '|' runs two at once: A|B")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--long", action="store_true", help="follow-up turns ask for a long answer (a long decode)")
    a = ap.parse_args()
    convs = {}
    for step in a.plan.split(","):
        names = step.split("|")
        for n in names:
            if n not in convs:
                convs[n] = Conv(n, 1000 + ord(n), a.words, a.long)
        if len(names) == 1:
            n = names[0]
            report(step, n, *convs[n].step(a.url, a.max_tokens))
            continue
        results = {}

        def run(n):
            results[n] = convs[n].step(a.url, a.max_tokens)

        ts = []
        for n in names:
            t = threading.Thread(target=run, args=(n,))
            t.start()
            ts.append(t)
            time.sleep(0.5)      # keep the order: the first holds the engine, the rest queue behind it
        for t in ts:
            t.join()
        for n in names:
            report(step, n, *results[n])


if __name__ == "__main__":
    sys.exit(main())
