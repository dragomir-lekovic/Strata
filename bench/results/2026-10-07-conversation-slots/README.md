# 2026-10-07: two conversations on the GPU (`--conversation-vram-slots 2`)

Question: two people share one engine at 128K context. Each turn of one used to cost the other a re-read of its whole
prompt. What does a switch cost with the RAM conversation cache, and with the standby VRAM slot on top of it?

Setup: RTX 5070 Ti 16 GB, Threadripper 7960X, 125 GB RAM; IQ3_S, `--max-context 131072 --kv int8 --kv-resident 32768`
(KV streaming), MTP `--spec 4`, the user's own `strata-iq3_s.json` otherwise. Three configs:
- *no cache*: that config as it is;
- *RAM cache*: `+ --conversation-cache-mib 32768 --conversation-cache-slots 64`;
- *slots*: the same `+ --conversation-vram-slots 2`.

`conv_switch.py` builds chats of ~120K tokens (deterministic word lists), thinking off, greedy, and asks each one a
short follow-up per turn. `prompt_ms` is the engine's time from the request to its first verify window: the switch
plus reading the new message (~30-45 tokens).

## Two users taking turns (`--plan A,B,A,B,A,B`)

| Config | switched turn (`prompt_ms`) | what the switch is |
|---|---:|---|
| no cache | 39,846 - 39,908 ms | the whole prompt again |
| RAM cache | 667 - 1,301 ms | park (301 - 940 ms) + restore (117 - 122 ms) |
| slots | **301 - 385 ms** | swap, 55 - 67 ms |

The rest of each slots figure is reading the new message (and, for a checkpoint resume, restoring the 118 MB
checkpoint); the switch itself is the 55 - 67 ms swap.

## Three users, each turn queued behind the previous one (`--plan A,B,C,A|B,C|A,B|C,A,B`)

`X|Y`: X runs, Y is sent 0.5 s later and waits; the server sends the engine `HINT` for Y. With three chats only two
fit on the GPU, so the queued one is usually in RAM: the slot worker writes the standby conversation back and loads
the queued one WHILE X decodes (248 - 304 ms, off the critical path), and Y's switch is a swap.

| | slots | RAM cache |
|---|---:|---:|
| queued turn (Y), prefetched | **349 - 566 ms** | 679 - 752 ms |
| turn nobody queued behind (X), from RAM | 975 - 1,258 ms | 705 - 1,312 ms |

The second row is no better than the RAM cache: without a hint the write-back + load + swap happen in front of the
request (as park + restore do). The first write-back of each conversation is a full capture; later ones append to the
retained K/V (`reused_kv_bytes` ~1.84 GB). In this run the prefetch write-backs took 615 - 1,128 ms: the slot worker
had inherited the host loop's pin to its core and shared it with the decode's spin - fixed (it widens its affinity),
and the runs below are after the fix (268 - 416 ms).

### Does the prefetch slow the decode it overlaps?

Static expert residency (`--adapt-every 100000`) makes both runs produce the same text, so tokens/s compare request
for request: the same nine requests once queued (`A,B,C,A|B,C|A,B|C`, the slot worker prefetches during A, C and B)
and once one after the other (`A,B,C,A,B,C,A,B,C`, no hints, every switch in the foreground).

| Request | decode, prefetch overlapping | decode, no prefetch | next request's `prompt_ms`, prefetched | same, not |
|---|---:|---:|---:|---:|
| A (turn 2) | 65.1 tok/s | 65.9 tok/s | B: 391 ms | 1,313 ms |
| C (turn 2) | 71.4 tok/s | 72.0 tok/s | A: 334 ms | 766 ms |
| B (turn 3) | 112.8 tok/s | 111.5 tok/s | C: 326 ms | 762 ms |

Within +-1.2%, in both directions: no slowdown the measurement can see.

## Exactness

`--adapt-every 100000` (static expert residency) and `STRATA_STATE_HASH=1`, `--words 3000` (~7.3K tokens):
A's three turns alone, against A, B, A, B, C, A - A's second turn through a swap, its third through write-back to RAM,
load into the slot and swap. Identical text and identical GDN, PLE, indexer, pooled, K/V and draft fingerprints after
every one of A's turns (`exactness-state-hashes.txt`).

## The swap's own cost

1.58 GiB of pinned K/V each way at 120K tokens, exchanged by CPU threads; the VRAM part (289 MiB) finishes on the GPU
in its shadow. The first version took 190 ms: its threads inherited the host loop's pin to one core
(SessionLoopScratch), so sixteen threads shared it. Widened to the process's start-up affinity they take 55 - 67 ms;
the same exchange in a standalone program takes 50 ms at 16 - 48 threads (memory-bound).
