#!/usr/bin/env python
"""
ppl_recheck.py (v3) — Recompute ONLY the perplexity column for MPAC and
BiMark (each with its own length-matched No-Embedding base columns),
replicating compare.py's trial logic
VERBATIM — same truncate-first protocol, same full-softmax scoring, same
seeding — with exactly ONE change:

    NUCLEUS FIX (MPAC & BiMark only): model.generate() silently merges the
    hub checkpoint's generation_config with the passed kwargs. Llama-3.1
    ships top_p=0.9, so a hidden nucleus warper stacked on top of the
    watermark processors, depressing Llama MPAC/BiMark PPL below the
    unwatermarked baseline. Fixed by passing top_p=1.0 explicitly and
    neutralizing the hub generation_config at load. Nothing else differs.

Kept identical to compare.py (do NOT "fix" these):
  * TopKTruncationProcessor(TOP_K) runs BEFORE the MPAC bias, so MPAC
    operates on the top-50-renormalized base distribution — the shared
    support of the whole comparison (BiMark restricts internally).
  * StealthInk is EXCLUDED: its custom sampling loop never merges the hub
    generation_config, so its table numbers are unaffected by this bug.
  * Scoring: full softmax for every scheme (compare.py's
    score_tokens_teacher_forced), no truncated reference, no clamps.
  * Seeding: payload = i % 256, trial_seed = seed_base + i*10 + n_tok, seed
    bases 70000/80000/90000.

Diagnostic column `oot50` = fraction of generated tokens outside the base
model's top-50 (recomputed in float32). Under the truncate-first protocol it
should be ~0 for EVERY scheme (a handful of 1e-3 events are fp16-vs-fp32
rank-50 ties, harmless). If a scheme shows oot50 >> 0, the protocol did not
replicate — trust that flag over the PPL numbers.

Output format matches compare.py's CSV perplexity block, per scheme row:
  ppl_wm_mean, ppl_wm_std, ppl_wm_se, ppl_base_mean, ppl_base_std,
  ppl_base_se, ppl_delta   (+ oot50_wm / oot50_base diagnostics appended).
Statistics use compare.py's _mean_std_sem convention (population std,
se = std/sqrt(n)). Per trial, order matches run_fixed_scheme exactly:
seed -> watermarked generation -> teacher-forced full-softmax scoring ->
length-matched base generation (no reseed; RNG state continues), so each
scheme row carries its own base columns, as in compare.py.

Block lengths: sweeps FIXED_NS = [20, 30, 40, 50, 60] (compare.py's
FIXED_NS), one CSV row per (model, scheme, n_tok); trial_seed and payload
match run_fixed_scheme at every (scheme, n_tok, trial).

Usage:
  python ppl_recheck.py                                   # 1000 trials (default)
  PPL_MODELS=llama PPL_TRIALS=5 PPL_NS=50 python ppl_recheck.py   # smoke test
Writes ppl_recheck.csv.
"""
from __future__ import annotations

import gc
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from datasets import load_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:          # so ./baselines/ is importable
    sys.path.insert(0, _THIS_DIR)

from baselines.mpac import WatermarkLogitsProcessor as MpacLogitsProcessor
from baselines.bimark import WatermarkBimark

# ── Config (mirrors compare.py; keep in sync) ───────────────────────────────
MODEL_NAMES = [
    "unsloth/Meta-Llama-3.1-8B",
    "unsloth/Qwen3.5-9B-Base",
    "unsloth/mistral-7b-v0.3",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRIALS  = int(os.environ.get("PPL_TRIALS", "1000"))
MODEL_SUB = os.environ.get("PPL_MODELS", "").lower()   # substring filter
FIXED_NS  = [int(x) for x in
             os.environ.get("PPL_NS", "20,30,40,50,60").split(",")]
K_BITS    = 8
TOP_K     = 50
N_PROMPTS = 200
PROMPT_TOKEN_LEN = 32
SHARED_SEED = 0x9E3779B97F4A7C15F39CC0605CEDC834

# MPAC knobs (same as compare.py)
MPAC_GAMMA, MPAC_DELTA, MPAC_RADIX, MPAC_SEEDING = 0.25, 1.5, 4, "simple_1"
# BiMark knobs (repo generate-script defaults, seeds derived like compare.py)
BIMARK_DELTA, BIMARK_LAYERS, BIMARK_WINDOW, BIMARK_TOPK = 0.2, 20, 2, 50
BIMARK_C_KEY, BIMARK_BIT_IDX_KEY = 8214793, 283519
BIMARK_PARTITION_SEEDS = [
    int(x) for x in
    np.random.default_rng(SHARED_SEED).choice(10000, BIMARK_LAYERS, replace=False)
]
# per-scheme trial seed bases — MUST match compare.py's FIXED_SCHEME_TABLE
SEED_BASE = {"MPAC": 70000, "BiMark": 80000}   # as compare.py FIXED_SCHEME_TABLE

OUT_CSV = "ppl_recheck.csv"


def log(*a, **kw):
    print(*a, **kw, flush=True)


# ── Copied VERBATIM from compare.py ─────────────────────────────────────────
class TopKTruncationProcessor:
    """Truncate logits to the top-k BEFORE a downstream watermark processor.

    Placed first in the logits_processor list, this hands the watermark
    scheme the top-k-renormalized base distribution, so MPAC biases within
    the same candidate set as the shared top-50 base reference.
    NOTE: truncating AFTER the watermark processor would instead re-distort
    the watermarked distribution, so order matters.
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, input_ids, scores):
        if self.k and 0 < self.k < scores.shape[-1]:
            kth = torch.topk(scores, self.k, dim=-1).values[..., -1, None]
            return scores.masked_fill(scores < kth, float("-inf"))
        return scores


# ── Model / prompts ─────────────────────────────────────────────────────────
def load_model(model_name):
    log(f"Loading {model_name} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(model_name,
                                        clean_up_tokenization_spaces=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()
    gcfg = getattr(model, "generation_config", None)
    if gcfg is not None:
        # NUCLEUS FIX (the only behavioral change vs the table run): log what
        # the checkpoint shipped, then neutralize every sampling field.
        log(f"  hub generation_config: temperature={gcfg.temperature} "
            f"top_p={gcfg.top_p} top_k={gcfg.top_k} do_sample={gcfg.do_sample}")
        gcfg.max_length = None
        gcfg.temperature = None
        gcfg.top_p = None
        gcfg.top_k = None
        gcfg.typical_p = None
        gcfg.epsilon_cutoff = None
        gcfg.eta_cutoff = None
    return tok, model


def build_prompt_pool(tokenizer):
    log(f"Loading {N_PROMPTS} C4 RealNews prompts ({PROMPT_TOKEN_LEN} tokens)...")
    pool = []
    try:
        ds = load_dataset("allenai/c4", "realnewslike", split="train",
                          streaming=True)
        ds = ds.shuffle(seed=12345, buffer_size=2000)   # same seed as compare.py
        for ex in ds:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            if len(ids) >= PROMPT_TOKEN_LEN:
                pool.append(ids[:PROMPT_TOKEN_LEN])
            if len(pool) >= N_PROMPTS:
                break
    except Exception as e:
        log(f"  C4 streaming failed: {e!r} — falling back to fixed prompts.")
        fallbacks = [
            "The Federal Reserve announced on Wednesday that it would maintain interest rates near zero ",
            "Researchers at MIT have developed a new algorithm that can detect early signs of ",
            "After months of negotiations, the European Union finalized a new trade agreement with ",
            "Stock markets in Asia closed higher on Friday, led by gains in technology and energy ",
            "A major hurricane is expected to make landfall along the eastern seaboard later this week ",
        ]
        for txt in fallbacks * (N_PROMPTS // len(fallbacks) + 1):
            ids = tokenizer.encode(txt, add_special_tokens=False)
            pool.append(ids[:PROMPT_TOKEN_LEN] if len(ids) >= PROMPT_TOKEN_LEN
                        else ids + [tokenizer.eos_token_id] * (PROMPT_TOKEN_LEN - len(ids)))
            if len(pool) >= N_PROMPTS:
                break
    log(f"  loaded {len(pool)} prompts")
    return pool


# ── Scoring: FULL softmax (compare.py's reference) + oot50 sanity check ─────
@torch.no_grad()
def score_full_softmax(model, prompt_ids, gen_ids):
    """Return (per-token full-softmax logps, #tokens outside base top-K)."""
    if not gen_ids:
        return [], 0
    full = list(prompt_ids) + list(gen_ids)
    ids = torch.tensor(full, dtype=torch.long,
                       device=model.device).unsqueeze(0)
    logits = model(ids, use_cache=False).logits[0].float()
    p_len = len(prompt_ids)
    logprobs = torch.log_softmax(logits, dim=-1)
    logps, oot = [], 0
    for i, tok in enumerate(gen_ids):
        pos = p_len + i - 1
        logps.append(float(logprobs[pos, tok].item()))
        if tok not in torch.topk(logits[pos], TOP_K).indices:
            oot += 1
    return logps, oot


def ppl(logps):
    return math.exp(-sum(logps) / len(logps)) if logps else float("nan")


# ── Generations: VERBATIM compare.py trial logic (+ nucleus fix where noted) ─
@torch.no_grad()
def gen_base(model, prompt_ids, n_tokens):
    """Length-matched no-embedding reference, exactly compare.py's
    baseline_logprobs_and_tokens: sample from the top-50-truncated
    distribution, but log the FULL-softmax probability of each sampled token
    (log probs[tok], not trunc[tok]). Returns (logps, token_ids)."""
    ids = torch.tensor(prompt_ids, dtype=torch.long,
                       device=model.device).unsqueeze(0)
    past, cur = None, ids
    logps, out_ids = [], []
    for _ in range(n_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        topk = torch.topk(probs, TOP_K)
        trunc = torch.zeros_like(probs)
        trunc[topk.indices] = topk.values
        trunc = trunc / trunc.sum()
        tok = int(torch.multinomial(trunc, 1).item())
        logps.append(float(torch.log(probs[tok].clamp_min(1e-30)).item()))
        out_ids.append(tok)
        cur = torch.tensor([[tok]], dtype=torch.long, device=model.device)
    return logps, out_ids


@torch.no_grad()
def gen_mpac(model, tokenizer, vocab_size, prompt_ids, payload_int, n_tokens):
    binary_msg = format(payload_int, f"0{K_BITS}b")
    proc = MpacLogitsProcessor(
        vocab=list(range(vocab_size)), gamma=MPAC_GAMMA, delta=MPAC_DELTA,
        seeding_scheme=MPAC_SEEDING, base=MPAC_RADIX,
        message_length=K_BITS, code_length=K_BITS, device=model.device)
    proc.set_message(binary_msg)
    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    output = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        do_sample=True,
        temperature=1.0,
        top_k=0,             # truncation handled by the processor below
        top_p=1.0,           # << NUCLEUS FIX: never inherit hub top_p
        min_new_tokens=n_tokens,
        max_new_tokens=n_tokens,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
        # Truncate-first, then bias (verbatim compare.py protocol).
        logits_processor=[TopKTruncationProcessor(TOP_K), proc],
    )
    return output[0, len(prompt_ids):].detach().cpu().tolist()


@torch.no_grad()
def gen_bimark(model, tokenizer, vocab_size, prompt_ids, payload_int, n_tokens):
    bits = format(payload_int, f"0{K_BITS}b")
    proc = WatermarkBimark(
        tokenizer=tokenizer, vocab_size=vocab_size, device=model.device,
        top_k=BIMARK_TOPK, partition_seeds=list(BIMARK_PARTITION_SEEDS),
        c_key=BIMARK_C_KEY, bit_idx_key=BIMARK_BIT_IDX_KEY,
        delta=BIMARK_DELTA, window_size=BIMARK_WINDOW, bits=bits)
    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    output = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        do_sample=True,
        temperature=1.0,
        top_k=0,             # processor already restricts to its top-50
        top_p=1.0,           # << NUCLEUS FIX: never inherit hub top_p
        min_new_tokens=n_tokens,
        max_new_tokens=n_tokens,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=[proc],
    )
    return output[0, len(prompt_ids):].detach().cpu().tolist()


# ── Driver ──────────────────────────────────────────────────────────────────
def mean_std_sem(vals):
    """Exactly compare.py's _mean_std_sem: population std, se = sd/sqrt(n)."""
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v)); sd = float(np.std(v))
    return m, sd, sd / math.sqrt(len(v))


def run_model(model_name, rows):
    tok, model = load_model(model_name)
    vocab_size = model.config.vocab_size
    pool = build_prompt_pool(tok)

    schemes = [
        ("MPAC",       SEED_BASE["MPAC"],
         lambda p, m, n: gen_mpac(model, tok, vocab_size, p, m, n)),
        ("BiMark",     SEED_BASE["BiMark"],
         lambda p, m, n: gen_bimark(model, tok, vocab_size, p, m, n)),
    ]

    # scheme-major, n_tok inner: identical order to run_fixed_scheme
    for name, seed_base, gen_fn in schemes:
      for n_tok in FIXED_NS:
        log("\n" + "=" * 72)
        log(f"[{model_name}] {name} n={n_tok}  — {N_TRIALS} trials")
        log("=" * 72)
        ppl_wm_vals, ppl_base_vals = [], []
        oot_wm = oot_base = ntok_wm = ntok_base = 0
        for i in range(N_TRIALS):
            prompt_ids = pool[i % len(pool)]
            payload = i % (2 ** K_BITS)
            trial_seed = seed_base + i * 10 + n_tok
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            # (1) watermarked generation
            gen_ids = gen_fn(prompt_ids, payload, n_tok)
            # (2) teacher-forced full-softmax scoring of the wm text
            logps_wm, o_wm = score_full_softmax(model, prompt_ids, gen_ids)
            ppl_wm = ppl(logps_wm)
            # (3) length-matched base generation — NO reseed, RNG state
            #     continues after the wm generation, exactly as compare.py
            bl, base_ids = gen_base(model, prompt_ids, len(gen_ids))
            ppl_b = ppl(bl)
            _, o_b = score_full_softmax(model, prompt_ids, base_ids)

            ppl_wm_vals.append(ppl_wm)
            ppl_base_vals.append(ppl_b)
            oot_wm += o_wm;  ntok_wm += len(gen_ids)
            oot_base += o_b; ntok_base += len(base_ids)
            if (i + 1) % 25 == 0 or i == 0:
                log(f"  trial {i + 1:>4}/{N_TRIALS}: "
                    f"ppl_wm={ppl_wm:.2f} ppl_base={ppl_b:.2f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        wm_m, wm_sd, wm_se = mean_std_sem(ppl_wm_vals)
        b_m, b_sd, b_se = mean_std_sem(ppl_base_vals)
        log(f"\n[{model_name}] {name} n={n_tok}: "
            f"ppl_wm={wm_m:.2f}±{wm_se:.2f} (sd {wm_sd:.2f})  "
            f"ppl_base={b_m:.2f}±{b_se:.2f} (sd {b_sd:.2f})  "
            f"delta={wm_m - b_m:+.2f}  "
            f"oot50_wm={oot_wm / max(1, ntok_wm):.4f} "
            f"oot50_base={oot_base / max(1, ntok_base):.4f}")
        rows.append(dict(model=model_name, scheme=name, n_tok=n_tok,
                         trials=N_TRIALS,
                         ppl_wm_mean=wm_m, ppl_wm_std=wm_sd, ppl_wm_se=wm_se,
                         ppl_base_mean=b_m, ppl_base_std=b_sd, ppl_base_se=b_se,
                         ppl_delta=wm_m - b_m,
                         oot50_wm=oot_wm / max(1, ntok_wm),
                         oot50_base=oot_base / max(1, ntok_base)))

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    rows = []
    for model_name in MODEL_NAMES:
        if MODEL_SUB and MODEL_SUB not in model_name.lower():
            continue
        log("\n" + "#" * 72)
        log(f"# MODEL: {model_name}")
        log("#" * 72)
        try:
            run_model(model_name, rows)
        except Exception as e:
            log(f"!! Model {model_name} failed: {e!r}")

    with open(OUT_CSV, "w") as f:
        f.write("model,scheme,n_tok,trials,"
                "ppl_wm_mean,ppl_wm_std,ppl_wm_se,"
                "ppl_base_mean,ppl_base_std,ppl_base_se,ppl_delta,"
                "oot50_wm,oot50_base\n")
        for r in rows:
            f.write(f"{r['model']},{r['scheme']},{r['n_tok']},{r['trials']},"
                    f"{r['ppl_wm_mean']:.4f},{r['ppl_wm_std']:.4f},"
                    f"{r['ppl_wm_se']:.4f},"
                    f"{r['ppl_base_mean']:.4f},{r['ppl_base_std']:.4f},"
                    f"{r['ppl_base_se']:.4f},{r['ppl_delta']:.4f},"
                    f"{r['oot50_wm']:.6f},{r['oot50_base']:.6f}\n")
    log(f"\nWrote {OUT_CSV}")
    log("Sanity: oot50 ~ 0 everywhere (truncate-first protocol; ~1e-3 fp16 "
        "rank-50 ties are fine). On Llama, MPAC/BiMark should move UP from "
        "~6.2 toward/above their base columns; Qwen/Mistral should "
        "reproduce the table within noise.")


if __name__ == "__main__":
    main()