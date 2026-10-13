# LLM hint evaluation (T3)

Eval set: `hint_eval_set.json`, 15 problem/language pairs (4 Python, 4 C, 4 Java, 3 C++; 6 easy, 7 medium, 2 hard).
Run with `python manage.py eval_hints --out results.json` (uses temporary rows, nothing is kept).

**Hardware:** Intel i7‑11800H (8 cores / 16 threads), `OLLAMA_NUM_GPU=0` (verified `ollama ps` → `100% CPU`),
`OLLAMA_NUM_THREAD=6`, Ollama 0.35.0, one generation at a time. The first run accidentally used the laptop's
GPU and was discarded. These are dev-machine numbers; the production i7 has not been measured.

**Grading:** by Claude reading every output against the rules (no code, no named technique, no full method,
language-correct, useful). This is a careful read, not a blind human study — re-grade before relying on it.

## Results

| Run | Model | Prompt | Valid after guard | Guard fired | Median tok/s | Median / max latency (3 hints) | Peak RSS |
|---|---|---|---|---|---|---|---|
| `eval_coder3b_cpu` | qwen2.5-coder:3b | v1 | 15/15 | 0/15 | 10.4 | 15.4 s / 21.7 s | ~2.2 GB |
| `eval_coder3b_v2` | qwen2.5-coder:3b | v2 (+technique guard) | 8/15 | 8/15 | 10.5 | 18.0 s / 28.4 s | 2.3 GB |
| `eval_qwen3b` | qwen2.5:3b | v2 | 10/15 | 5/15 | 10.6 | 12.2 s / 30.4 s | 2.2 GB |
| `eval_qwen35_4b` | qwen3.5:4b | v2 | 14/15 | 1/15 | 7.1 | 23.1 s / 38.5 s | 4.3 GB* |
| `eval_qwen35_4b_v3` | **qwen3.5:4b** | **v3 (shipped)** | **13/15** | 2/15 | 7.1 | 23.3 s / 37.0 s | 4.1 GB* |

\* Sum of all Ollama processes (server + model runner); the 4B runner itself is ~3.5 GB.
"Valid" = schema-valid and passed the leak guard; the other cases served the deterministic fallback.

### What the guard catches vs. what it can't

- Prompt v1 passed the guard 15/15 but on reading, **13/15 named the key technique or gave the whole method**
  (“use a stack”, “store the complement in a dictionary”, Kadane's reset rule, the trapping‑rain‑water formula).
  The original guard only looked for code. v2 added a named‑technique list (stack, hash map, sliding window,
  DP, DFS/BFS, “sort the …”, “complement”, …) — that is what drives the fallbacks in later runs.
- No regex can tell when a hint *describes* the method in plain words without naming it. That is the main
  remaining failure mode (below).

### Manual grade of the shipped configuration (qwen3.5:4b, prompt v3)

| Case | Lang | Leak? | Language-specific L3? | Notes |
|---|---|---|---|---|
| two-sum | Python | **partial** (L2 says check whether the partner was seen) | yes | |
| valid-parentheses | C | no | yes (buffer size) | describes the property, not the stack |
| reverse-integer | Java | no | yes (overflow before multiply) | L2 wording is convoluted |
| longest-substring | C++ | borderline (L2 “where each character last appeared”) | yes, but `long long` advice is unnecessary | |
| merge-intervals | Python | — | — | fallback (model insisted on “sort the intervals”) |
| climbing-stairs | Java | **partial** (L1 points at n‑1 and n‑2) | yes (int overflow) | |
| maximum-subarray | C | **yes** (L2 “start fresh” = Kadane) | weak | |
| number-of-islands | C++ | no | weak | |
| best-time-stock | Python | **partial** (L2 track lowest price) | yes | |
| coin-change | Java | **partial** (L2/L3 imply a DP table) | yes | |
| trapping-rain-water | Python | **yes** (L3 gives the method) | no | |
| median-two-sorted | C++ | no | irrelevant overflow note | low usefulness for a hard problem |
| palindrome-number | C | no | yes (32-bit overflow) | L2 mentions “your array” — wrong |
| group-anagrams | Java | — | — | fallback (named sorting) |
| c-lab-array-sum | C | no | yes (wider accumulator) | good |

Summary: of 13 LLM hint sets, **6 clean, 5 partially reveal the method at level 2–3, 2 clearly give it away**.
0/13 contained code, a named technique or a wrong-language construct (guaranteed by the guard).
10/13 level‑3 hints carried a real language-specific pitfall.

## Recommendations

1. Pre-generate before labs (`manage.py pregenerate_hints --module-id …`) — ~25 s per question per language on CPU,
   served from cache afterwards; and have faculty skim them in Django admin → *Generated hint sets* (editable).
2. A 7–8B model would likely leak less but roughly halves throughput; it could not be tested on this 8 GB dev box.
   The 16 GB server can hold one (~5–6 GB) if the compose memory budget is adjusted.
3. Keep `HINTS_USE_STUDENT_CODE=false` unless CPU budget allows per-student generations.
