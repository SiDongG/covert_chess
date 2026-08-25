#!/usr/bin/env python
"""
ppl_recheck.py — Recompute ONLY the perplexity column for the questionable
schemes (MPAC, BiMark, StealthInk) plus the No-Embedding reference.

What it corrects relative to the run that produced the current table:

  (1) GENERATION (the actual bug, MPAC & BiMark only): model.generate()
      silently merges the hub checkpoint's generation_config with the passed
      kwargs. Llama-3.1 checkpoints ship top_p=0.9, so on Llama a hidden
      nucleus warper stacked on top of the watermark processors, depressing
      MPAC/BiMark PPL ~19% below the unwatermarked baseline (impossible for
      any honest watermark). Fixed here by neutralizing the hub
      generation_config at load AND passing top_p=1.0 explicitly.
      StealthInk uses its own sampling loop and never had this problem; it is
      rerun unchanged as a control and to confirm its column was clean.

  (2) SCORING (a landmine in the current compare.py, not in the table run):
      all schemes are teacher-forced under the base model's FULL softmax —
      the same reference the base / BAM / ArcMark / BiMark rows already use.
      No top-k truncation, no 1e-30 clamps.

Also reported per scheme: the fraction of generated tokens falling OUTSIDE
the base model's top-50 (`oot50`). Expectations:
  - base:      0 by construction (sampled from top-50)
  - BiMark:    0 (its processor restricts to the base top-50 internally)
  - MPAC:      > 0 (full-vocab biased sampling — proves no hidden truncation)
  - StealthInk:> 0 (full-vocab reweighted sampling, repo default)
If MPAC/BiMark oot50 or PPL still look truncated on some model, the printed
`hub generation_config` line for that model tells you what else shipped.

Seeding replicates run_fixed_scheme exactly (payload = i % 256, trial_seed =
seed_base + i*10 + n_tok, same per-scheme seed bases), so StealthInk's
generations are bit-identical to the original run; MPAC/BiMark differ only
through the corrected warper stack.

Usage:
  python ppl_recheck.py                     # all models, 200 trials, n=50
  PPL_TRIALS=1000 python ppl_recheck.py     # match the paper's trial count
  PPL_MODELS=llama python ppl_recheck.py    # substring filter on model name

Writes ppl_recheck.csv and prints a summary table.
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
from baselines.stealthink import (ReweightProcessor as SIReweightProcessor,
                                  ReweightLogitsProcessor as SIReweightLogitsProcessor,
                                  generate_exact_n_tokens as si_generate_exact_n_tokens)

# ── Config (mirrors compare.py; keep in sync) ───────────────────────────────
MODEL_NAMES = [
    "unsloth/Meta-Llama-3.1-8B",
    #"unsloth/Qwen3.5-9B-Base",
    #"unsloth/mistral-7b-v0.3",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRIALS  = int(os.environ.get("PPL_TRIALS", "200"))
MODEL_SUB = os.environ.get("PPL_MODELS", "").lower()   # substring filter
N_TOK     = 50                                          # the table's row
K_BITS    = 8
M_MSG     = 256
TOP_K     = 50          # base sampler truncation + oot diagnostic threshold
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
# StealthInk knobs
STEALTHINK_CAPACITY, STEALTHINK_NGRAM = 1, 3

# per-scheme trial seed bases — MUST match compare.py's FIXED_SCHEME_TABLE
SEED_BASE = {"MPAC": 70000, "BiMark": 80000, "StealthInk": 90000}

OUT_CSV = "ppl_recheck.csv"


def log(*a, **kw):
    print(*a, **kw, flush=True)


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
        # THE FIX: hub checkpoints ship sampling defaults that generate()
        # silently merges with our kwargs (Llama-3.1 ships top_p=0.9). Log
        # what shipped, then neutralize everything.
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


# ── Scoring: FULL softmax + out-of-top-50 diagnostic in one forward pass ────
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
        topk_idx = torch.topk(logits[pos], TOP_K).indices
        if tok not in topk_idx:
            oot += 1
    return logps, oot


def ppl(logps):
    return math.exp(-sum(logps) / len(logps)) if logps else float("nan")


# ── Generations ─────────────────────────────────────────────────────────────
@torch.no_grad()
def gen_base(model, prompt_ids, n_tokens):
    """No-embedding reference: top-50 sampling (ArcMark cover), returns ids."""
    ids = torch.tensor(prompt_ids, dtype=torch.long,
                       device=model.device).unsqueeze(0)
    past = None
    cur = ids
    out_ids = []
    for _ in range(n_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        topk = torch.topk(probs, TOP_K)
        trunc = torch.zeros_like(probs)
        trunc[topk.indices] = topk.values
        trunc = trunc / trunc.sum()
        tok = int(torch.multinomial(trunc, 1).item())
        out_ids.append(tok)
        cur = torch.tensor([[tok]], dtype=torch.long, device=model.device)
    return out_ids


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
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        do_sample=True, temperature=1.0,
        top_k=0,
        top_p=1.0,               # << the fix: never inherit hub top_p
        min_new_tokens=n_tokens, max_new_tokens=n_tokens,
        eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
        logits_processor=[proc])
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
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        do_sample=True, temperature=1.0,
        top_k=0,
        top_p=1.0,               # << the fix
        min_new_tokens=n_tokens, max_new_tokens=n_tokens,
        eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
        logits_processor=[proc])
    return output[0, len(prompt_ids):].detach().cpu().tolist()


@torch.no_grad()
def gen_stealthink(model, tokenizer, vocab_size, prompt_ids, payload_int,
                   n_tokens):
    """Unchanged from compare.py — custom loop, never touched by the bug."""
    num_value = 2 ** STEALTHINK_CAPACITY
    R = 1.0 / num_value
    converted_msg_length = K_BITS // STEALTHINK_CAPACITY
    bits = format(payload_int, f"0{K_BITS}b")
    embedded_message = [
        int(bits[p * STEALTHINK_CAPACITY:(p + 1) * STEALTHINK_CAPACITY], 2)
        for p in range(converted_msg_length)]
    vocab = list(range(vocab_size))       # logits dim, matching compare.py
    rp = SIReweightProcessor(vocab=vocab)
    lp = SIReweightLogitsProcessor(
        rp, embedded_message=embedded_message, n_gram_len=STEALTHINK_NGRAM,
        R=R, converted_msg_length=converted_msg_length, seen_seeds=set())
    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    seq = si_generate_exact_n_tokens(
        model=model, tokenizer=tokenizer, inputs=inputs,
        logits_processor=LogitsProcessorList([lp]),
        n_new_tokens=n_tokens, do_sample=True, temperature=1.0, top_k=0,
        eos_id=tokenizer.eos_token_id, soft_eos_penalty=0.0)
    return seq[0, len(prompt_ids):].detach().cpu().tolist()


# ── Driver ──────────────────────────────────────────────────────────────────
def mean_se(vals):
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v) / math.sqrt(len(v)))


def run_model(model_name, rows):
    tok, model = load_model(model_name)
    vocab_size = model.config.vocab_size
    pool = build_prompt_pool(tok)

    schemes = {
        "No Embedding": None,   # generated fresh per trial for reference
        "MPAC":       lambda p, m: gen_mpac(model, tok, vocab_size, p, m, N_TOK),
        "BiMark":     lambda p, m: gen_bimark(model, tok, vocab_size, p, m, N_TOK),
        "StealthInk": lambda p, m: gen_stealthink(model, tok, vocab_size, p, m, N_TOK),
    }

    results = {s: {"ppl": [], "oot": 0, "ntok": 0} for s in schemes}
    for i in range(N_TRIALS):
        prompt_ids = pool[i % len(pool)]
        payload = i % (2 ** K_BITS)         # identical to run_fixed_scheme

        for name, gen_fn in schemes.items():
            if gen_fn is None:
                # base reference: one seed convention, reused across schemes
                np.random.seed(50000 + i * 10 + N_TOK)
                torch.manual_seed(50000 + i * 10 + N_TOK)
                gen_ids = gen_base(model, prompt_ids, N_TOK)
            else:
                trial_seed = SEED_BASE[name] + i * 10 + N_TOK   # as compare.py
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                gen_ids = gen_fn(prompt_ids, payload)
            logps, oot = score_full_softmax(model, prompt_ids, gen_ids)
            results[name]["ppl"].append(ppl(logps))
            results[name]["oot"] += oot
            results[name]["ntok"] += len(gen_ids)

        if (i + 1) % 10 == 0 or i == 0:
            snap = "  ".join(f"{n}={np.nanmean(r['ppl']):.2f}"
                             for n, r in results.items())
            log(f"  trial {i + 1:>4}/{N_TRIALS}:  {snap}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\n[{model_name}] PPL recheck (n={N_TOK}, {N_TRIALS} trials, "
        f"full-softmax reference):")
    log(f"  {'scheme':<14} {'PPL mean±SE':>16} {'oot50 rate':>11}")
    for name, r in results.items():
        m, se = mean_se(r["ppl"])
        oot_rate = r["oot"] / max(1, r["ntok"])
        log(f"  {name:<14} {m:>10.2f} ± {se:.2f} {oot_rate:>10.4f}")
        rows.append(dict(model=model_name, scheme=name, n_tok=N_TOK,
                         trials=N_TRIALS, ppl_mean=m, ppl_se=se,
                         oot50_rate=oot_rate))

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
        f.write("model,scheme,n_tok,trials,ppl_mean,ppl_se,oot50_rate\n")
        for r in rows:
            f.write(f"{r['model']},{r['scheme']},{r['n_tok']},{r['trials']},"
                    f"{r['ppl_mean']:.4f},{r['ppl_se']:.4f},"
                    f"{r['oot50_rate']:.6f}\n")
    log(f"\nWrote {OUT_CSV}")
    log("Expected sanity pattern per model: base=reference; "
        "StealthInk≈base (distortion-free, oot50>0); "
        "BiMark≈base, oot50=0; MPAC above base, oot50>0.")


if __name__ == "__main__":
    main()