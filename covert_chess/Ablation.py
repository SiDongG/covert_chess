"""
ablation_confirmation.py — Confirmation-phase ablation on variable-length BAM.

Compares, on C4 RealNews, the length-vs-error frontier of THREE schemes, all
sharing the identical posterior-matching communication phase:

  1. BAM (two-phase, original):  communication phase with fixed decision
     threshold gamma=0.5, followed by the confirmation phase — posterior
     matching at p=2 over antipodal symbols u_ACK/u_NACK — with
     rho_ACK = 1 - 1/L (swept), rho_NACK = 0.75.

  2. ONE-PHASE (ablation):       NO confirmation phase. The communication
     phase runs until the message posterior itself crosses the threshold
     g = 1 - 1/L (same L sweep) and then COMMITS immediately to the argmax.

  3. BAM-FIXED-LENGTH (ablation): NO threshold decoding at all. The
     communication phase runs for a FIXED number of tokens
     n in {20, 30, 40, 50, 60, 70, 80} and then commits to the argmax of
     the message posterior. Sweeping n traces the fixed-length frontier.
     Since the token trajectory does not depend on n, a single run to
     max(n) yields the paired result at every checkpoint.

All schemes are run for an 8-bit payload (M=256), giving 3 frontier curves
per model.

Metrics kept deliberately minimal: message error rate and average token
count, each with a standard error (error: sqrt(p(1-p)/n); length:
std/sqrt(n)). Everything else from compare.py (perplexity, BER, distinct-n,
steganalysis, judge dumps) is removed.

Trials are PAIRED: for a given (K_BITS, trial index) the prompt, the true
message, and the RNG seeding are identical across the three schemes and
across all L / fixed-length values.
"""

from __future__ import annotations

import math
import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARCMARK_SRC = os.environ.get(
    "ARCMARK_SRC",
    os.path.normpath(os.path.join(_THIS_DIR, "..", "arcmark_src")),
)
if ARCMARK_SRC not in sys.path:
    sys.path.insert(0, ARCMARK_SRC)

from arcmark.config import ArcMarkConfig
from arcmark.sinkhorn import extract_conditional, solve_arcmark_ot
from arcmark.side_info import SideInfoMode, compute_key_si
from arcmark.coding import RandomLinearCode
from arcmark.processor import ArcMarkLogitsProcessor
from arcmark.message_decoder import decode_with_code, decode_message


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAMES = [
    #"unsloth/Meta-Llama-3.1-8B",
    #"unsloth/Qwen3.5-9B-Base",
    "unsloth/mistral-7b-v0.3",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False
N_TRIALS   = 2000 if not SMOKE_TEST else 4

# Payload size: 8-bit only (M=256) -> 3 schemes = 3 curves.
K_BITS_LIST = [8]

# The swept knob for the two variable-length schemes. For BAM it sets
# rho_ACK = 1 - 1/L; for the one-phase ablation it sets the commit
# threshold g = 1 - 1/L directly.
L_VALUES = (
    [2, 4, 8, 16, 32, 64, 128, 512, 2048]
    if not SMOKE_TEST else [8]
)

# The swept knob for the fixed-length ablation: commit to the posterior
# argmax after exactly n tokens, no stopping rule.
FIXED_LENGTHS = [20, 30, 40, 50, 60] if not SMOKE_TEST else [20]

# The swept knob for Scheme 4 (fixed-length ArcMark baseline): embed the
# payload as a length-n random-linear codeword and decode by codebook
# correlation. Swept over the SAME budgets as the BAM-fixed-length ablation so
# the two fixed-length curves are directly comparable at each token count.
FL_LENGTHS = FIXED_LENGTHS

GAMMA      = 0.5     # communication-phase decision threshold g1 (BAM only)
RHO_NACK   = 0.75    # rho_NACK (BAM only)
EPS_NOISE  = 0.4     # eps      (communication-phase Laplace floor)
EPS_CONF   = 0.4     # eps_ACK  (confirmation-phase antipodal floor, BAM only)

MAX_TOKENS      = 1000
MAX_CONF_STEPS  = 120
MIN_COMM_TOKENS = 0

OUT_PLOT = "ablation_confirmation.png"
OUT_CSV  = "ablation_confirmation.csv"

# ── Shared ArcMark core knobs ───────────────────────────────────────────────
P_FIELD            = 4
R_RESOLUTION       = 4
# 128-bit shared seed (lambda = 128). side_info.py encodes the seed as 16
# little-endian bytes so the full 128 bits flow into the SHA-256 that derives
# (s_index, perm_seed, R_t). Must match the constant used in compare.py.
SHARED_SEED        = 0x9E3779B97F4A7C15F39CC0605CEDC834
TOP_K              = 50
SINKHORN_REG       = 0.2
SINKHORN_MAX_ITER  = 4000
SINKHORN_STOP_THR  = 1e-4
PHI                = 0.0

# Seed for the fixed-length ArcMark random-linear codebook (Scheme 4). Masked
# to 32 bits so it is accepted regardless of the RNG RandomLinearCode.build
# uses internally (SHARED_SEED is 128-bit). Only self-consistency matters here:
# the decoder is handed the cached code OBJECT directly, so this seed never has
# to be re-derived at decode time. compare.py used SHARED_SEED + 42.
CODE_SEED = (SHARED_SEED + 42) & 0xFFFFFFFF

N_PROMPTS         = 200
PROMPT_TOKEN_LEN  = 32

ARC_CONFIG = ArcMarkConfig(
    top_k=TOP_K,
    top_p=None,
    sinkhorn_reg=SINKHORN_REG,
    max_iter=SINKHORN_MAX_ITER,
    stop_thr=SINKHORN_STOP_THR,
    min_tokens=2,
    method="sinkhorn_log",
    context_width=3,
    hash_keys=True,
)
SIDE_INFO_MODE = SideInfoMode.HASH_CONTEXT


def log(*a, **kw):
    print(*a, **kw, flush=True)


# ============================================================================
# Per-model context
# ============================================================================
class LMContext:
    def __init__(self, model_name: str):
        self.model_name = model_name
        log(f"Loading {model_name} on {DEVICE} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=DEVICE
        )
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        self.perm_cache: dict[int, torch.Tensor] = {}
        self.code_cache: dict = {}          # (n_tokens, n_msg) -> RandomLinearCode
        self.prompt_pool: list[list[int]] = []
        log(f"Loaded {model_name}. vocab_size={self.vocab_size}")

    def teardown(self):
        self.model = None
        self.tokenizer = None
        self.perm_cache.clear()
        self.code_cache.clear()
        self.prompt_pool = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


CTX: "LMContext | None" = None


class IncrementalLM:
    def __init__(self, prompt_ids: list[int]):
        model = CTX.model
        ids = torch.tensor(prompt_ids, dtype=torch.long,
                           device=model.device).unsqueeze(0)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        self.past = out.past_key_values
        self._last_logits = out.logits[0, -1].float()

    @torch.no_grad()
    def probs(self) -> torch.Tensor:
        return torch.softmax(self._last_logits, dim=-1)

    @torch.no_grad()
    def advance(self, token_id: int) -> None:
        model = CTX.model
        ids = torch.tensor([[token_id]], dtype=torch.long, device=model.device)
        out = model(ids, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self._last_logits = out.logits[0, -1].float()

    def free(self) -> None:
        self.past = None
        self._last_logits = None


# ============================================================================
# C4 RealNews prompts
# ============================================================================
def build_prompt_pool() -> list[list[int]]:
    tokenizer = CTX.tokenizer
    log(f"Loading {N_PROMPTS} C4 RealNews prompts ({PROMPT_TOKEN_LEN} tokens each)...")
    pool: list[list[int]] = []
    try:
        ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        ds = ds.shuffle(seed=12345, buffer_size=2000)
        for ex in ds:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            if len(ids) >= PROMPT_TOKEN_LEN:
                pool.append(ids[:PROMPT_TOKEN_LEN])
            if len(pool) >= N_PROMPTS:
                break
    except Exception as e:
        log(f"  C4 streaming failed: {e}")
        log("  Falling back to fixed C4-style prompts.")
        fallbacks = [
            "The Federal Reserve announced on Wednesday that it would maintain interest rates near zero ",
            "Researchers at MIT have developed a new algorithm that can detect early signs of ",
            "After months of negotiations, the European Union finalized a new trade agreement with ",
            "Stock markets in Asia closed higher on Friday, led by gains in technology and energy ",
            "A major hurricane is expected to make landfall along the eastern seaboard later this week ",
        ]
        for txt in fallbacks * (N_PROMPTS // len(fallbacks) + 1):
            ids = tokenizer.encode(txt, add_special_tokens=False)
            pool.append(
                ids[:PROMPT_TOKEN_LEN] if len(ids) >= PROMPT_TOKEN_LEN
                else ids + [tokenizer.eos_token_id] * (PROMPT_TOKEN_LEN - len(ids))
            )
            if len(pool) >= N_PROMPTS:
                break

    log(f"  loaded {len(pool)} prompts")
    log(f"  example prompt: {tokenizer.decode(pool[0])!r}")
    return pool


# ============================================================================
# Shared emission primitive
# ============================================================================
def _context_tokens_for_step(emitted: list[int], context_width: int) -> tuple[int, ...]:
    pad_len = max(0, context_width - len(emitted))
    return tuple([0] * pad_len + emitted[-context_width:])


def side_info_for_step(emitted: list[int]) -> tuple[int, int, float]:
    """Synchronized per-token side information (s_index, perm_seed, R_t).

    One keyed SHA-256 over (secret || context) split into three disjoint
    blocks: the channel index k_t^{(1)} (s_index), the vocabulary-permutation
    seed Lambda_t^{(2)} (perm_seed), and the posterior-matching randomness
    R_t = Rand(Lambda_t^{(3)}). Because it is derived from the shared seed and
    the shared transcript, encoder and decoder reconstruct the identical R_t;
    there is no unsynchronized local randomness.
    """
    context_tokens = _context_tokens_for_step(emitted, ARC_CONFIG.context_width)
    s_index, perm_seed, R_t = compute_key_si(
        secret_key=SHARED_SEED,
        context_tokens=context_tokens,
        num_keys=R_RESOLUTION,
        mode=SIDE_INFO_MODE,
        tokenizer=CTX.tokenizer,
        return_r=True,
    )
    return s_index, perm_seed, R_t


@torch.no_grad()
def emit_token(probs: torch.Tensor, emitted: list[int], symbol: int,
               alphabet_size: int = P_FIELD) -> int:
    s_index, perm_seed, _ = side_info_for_step(emitted)
    perm = _perm_for_seed(perm_seed, probs.device)

    ot_result = solve_arcmark_ot(
        probs,
        codeword_symbol=int(symbol),
        alphabet_size=int(alphabet_size),
        num_keys=R_RESOLUTION,
        vocab_size=CTX.vocab_size,
        perm=perm,
        phi=PHI,
        config=ARC_CONFIG,
    )
    cond = extract_conditional(
        ot_result.coupling,
        s_index,
        num_keys=R_RESOLUTION,
        full_vocab_size=CTX.vocab_size,
        token_indices=ot_result.token_indices,
    )
    token = int(torch.multinomial(cond, num_samples=1).item())
    return token


def _perm_for_seed(perm_seed: int, device) -> torch.Tensor:
    cache = CTX.perm_cache
    perm = cache.get(perm_seed)
    if perm is None or perm.device != device:
        from arcmark import geometry
        perm = geometry.random_permutation(CTX.vocab_size, seed=perm_seed).to(device)
        if len(cache) > 256:
            cache.clear()
        cache[perm_seed] = perm
    return perm


def read_symbol_angle(token_id: int, emitted_before: list[int]) -> float:
    s_index, perm_seed, _ = side_info_for_step(emitted_before)
    perm = _perm_for_seed(perm_seed, CTX.model.device)
    permuted_id = int(perm[token_id].item())
    theta = (2.0 * math.pi) * permuted_id / float(CTX.vocab_size)
    s_angle = (2.0 * math.pi) * s_index / float(R_RESOLUTION)
    return (theta - s_angle) % (2.0 * math.pi)


# ============================================================================
# Communication-phase likelihood (LAPLACE core, full 4-symbol)
# ============================================================================
P_SYM = P_FIELD


def posterior_match_symbol(pi: np.ndarray, m: int, R: float) -> int:
    V = float(pi[:m].sum() + R * pi[m])
    return min(int(P_SYM * V), P_SYM - 1)


_SIGMA     = math.pi / P_SYM
_B_LAPLACE = float(os.environ.get("LAPLACE_B", _SIGMA / math.sqrt(2.0)))
_Z_SIGNAL = 2.0 * _B_LAPLACE * (1.0 - math.exp(-math.pi / _B_LAPLACE))


def per_symbol_likelihood(angle_obs: float) -> np.ndarray:
    signal = np.empty(P_SYM)
    for u in range(P_SYM):
        target = (2 * math.pi * u / P_SYM + PHI) % (2 * math.pi)
        d = abs(angle_obs - target) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        signal[u] = math.exp(-d / _B_LAPLACE) / _Z_SIGNAL
    ells = (1.0 - EPS_NOISE) * signal + EPS_NOISE / (2.0 * math.pi)
    return ells


def message_likelihood(ells: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Vectorized version of compare.py's message_likelihood.

    The original per-message Python loop is O(M * P_SYM) in interpreted code,
    which is fine for M=256 but prohibitive at M=65536 (16-bit payloads) when
    called once per emitted token. This computes the same quantity with numpy
    array ops over the M message CDF intervals.
    """
    cdf = np.concatenate([[0.0], np.cumsum(pi)])
    lo = cdf[:-1]
    hi = cdf[1:]
    width = np.maximum(hi - lo, 1e-30)
    q = np.zeros(len(pi))
    for u in range(P_SYM):
        ul, uh = u / P_SYM, (u + 1) / P_SYM
        ov = np.maximum(0.0, np.minimum(hi, uh) - np.maximum(lo, ul))
        q += (ov / width) * ells[u]
    return q


def run_comm_step(pi, m_true, emitted, lm: "IncrementalLM"):
    # R_t: synchronized, transcript-derived posterior-matching randomness
    # Rand(Lambda_t^{(3)}), reconstructable by a black-box decoder — NOT a
    # local np.random draw.
    _, _, R = side_info_for_step(emitted)
    u = posterior_match_symbol(pi, m_true, R)
    probs = lm.probs()
    x = emit_token(probs, emitted, u)
    angle = read_symbol_angle(x, emitted)
    emitted.append(x)
    lm.advance(x)
    ells = per_symbol_likelihood(angle)
    q = message_likelihood(ells, pi)
    pi_new = pi * q
    s = pi_new.sum()
    pi_new = pi_new / s if s > 0 else np.ones_like(pi) / len(pi)
    return pi_new


# ============================================================================
# BAM confirmation phase — Algorithm 1 (posterior matching) at p = 2
# ============================================================================

P_CONF = 2                       # confirmation alphabet size (paper: p = 2)
SYM_ACK  = 0                     # u_ACK  -> angle 0
SYM_NACK = 1                     # u_NACK -> angle pi at p = 2

_B_CONF  = math.pi / (P_CONF * math.sqrt(2.0))
_Z_CONF  = 2.0 * _B_CONF * (1.0 - math.exp(-math.pi / _B_CONF))


def conf_symbol_likelihood(angle_obs: float) -> np.ndarray:
    """Contaminated-Laplace per-symbol likelihood at p = 2 over {u_ACK, u_NACK}.

    Same functional form as per_symbol_likelihood, instantiated with
    p = P_CONF = 2, the confirmation Laplace scale _B_CONF, and the
    confirmation contamination floor EPS_CONF (= eps_ACK).
    """
    signal = np.empty(P_CONF)
    for u in range(P_CONF):
        target = (2 * math.pi * u / P_CONF + PHI) % (2 * math.pi)
        d = abs(angle_obs - target) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        signal[u] = math.exp(-d / _B_CONF) / _Z_CONF
    return (1.0 - EPS_CONF) * signal + EPS_CONF / (2.0 * math.pi)


def run_confirmation(true_bit, emitted, lm: "IncrementalLM", max_steps,
                     g_ack, g_nack):
    """Confirmation phase = Algorithm 1 (posterior matching) at p = 2.

    true_bit selects the antipodal symbol (0 -> u_ACK angle 0; 1 -> u_NACK
    angle pi), emitted through a genuine p = 2 ArcMark OT channel; rho over
    {u_ACK, u_NACK} is updated with the p = 2 contaminated-Laplace likelihood
    (floor eps_ACK). Accept when rho[0] >= g_ack, reject when rho[1] >= g_nack.
    """
    tx_symbol = SYM_ACK if true_bit == 0 else SYM_NACK
    rho = np.array([0.5, 0.5])
    used = 0
    for _ in range(max_steps):
        probs = lm.probs()
        x = emit_token(probs, emitted, tx_symbol, alphabet_size=P_CONF)
        angle = read_symbol_angle(x, emitted)
        emitted.append(x)
        lm.advance(x)
        ell = conf_symbol_likelihood(angle)
        rho = rho * ell
        rho /= rho.sum()
        used += 1
        if rho[0] >= g_ack:  return "ACK",  used
        if rho[1] >= g_nack: return "NACK", used
    return ("ACK" if rho[0] >= rho[1] else "NACK"), used


# ============================================================================
# Scheme 1 — BAM (two-phase, original)
# ============================================================================
def bam_two_phase(prompt_ids, m_true, n_msg, max_tokens, g1, ra, rn,
                  min_comm=MIN_COMM_TOKENS):
    """Original BAM: comm phase with threshold g1=GAMMA, then antipodal
    confirmation with thresholds (ra=1-1/L, rn=RHO_NACK)."""
    pi = np.ones(n_msg) / n_msg
    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    knockdown = np.ones(n_msg)
    t = 0
    try:
        while t < max_tokens:
            pi = run_comm_step(pi, m_true, emitted, lm)
            eff = pi * knockdown; eff = eff / eff.sum()
            t += 1
            if t < min_comm:
                continue
            if eff.max() >= g1:
                cand = int(eff.argmax())
                true_bit = 0 if cand == m_true else 1
                outcome, ct = run_confirmation(
                    true_bit, emitted, lm, MAX_CONF_STEPS, ra, rn)
                t += ct
                if outcome == "ACK":
                    return cand == m_true, t, "ack"
                knockdown[cand] *= (1 - rn) / rn
                pi = pi * knockdown; pi /= pi.sum()
                knockdown = np.ones(n_msg)
                if t >= max_tokens:
                    break
        eff = pi * knockdown; eff /= eff.sum()
        decoded = int(eff.argmax())
        return decoded == m_true, t, "forced"
    finally:
        lm.free()


# ============================================================================
# Scheme 2 — ONE-PHASE ablation (no confirmation; threshold 1 - 1/L)
# ============================================================================
def one_phase_threshold(prompt_ids, m_true, n_msg, max_tokens, g,
                        min_comm=MIN_COMM_TOKENS):
    """Ablation: identical communication phase, but the commit threshold is
    g = 1 - 1/L applied DIRECTLY to the message posterior. The scheme commits
    to the argmax the moment any message crosses g — no confirmation phase,
    no knockdown/retry machinery."""
    pi = np.ones(n_msg) / n_msg
    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    t = 0
    try:
        while t < max_tokens:
            pi = run_comm_step(pi, m_true, emitted, lm)
            t += 1
            if t < min_comm:
                continue
            if pi.max() >= g:
                decoded = int(pi.argmax())
                return decoded == m_true, t, "commit"
        decoded = int(pi.argmax())
        return decoded == m_true, t, "forced"
    finally:
        lm.free()


# ============================================================================
# Scheme 3 — BAM-FIXED-LENGTH ablation (no threshold; commit at fixed n)
# ============================================================================
def bam_fixed_length(prompt_ids, m_true, n_msg, checkpoints):
    """Ablation: identical posterior-matching communication phase, but NO
    threshold decoding at all. The scheme runs for a FIXED number of tokens
    n and then commits to the argmax of the message posterior.

    Because there is no stopping rule, the token trajectory is independent
    of n: the run at n=20 is a prefix of the run at n=80 under the same
    seed. A single rollout to max(checkpoints) therefore yields the paired
    decode result at every fixed length, at 1/len(checkpoints) the cost.

    Returns {n: ok} for each n in checkpoints.
    """
    pi = np.ones(n_msg) / n_msg
    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    results: dict[int, bool] = {}
    t = 0
    try:
        for n_fix in sorted(checkpoints):
            while t < n_fix:
                pi = run_comm_step(pi, m_true, emitted, lm)
                t += 1
            results[n_fix] = (int(pi.argmax()) == m_true)
        return results
    finally:
        lm.free()


# ============================================================================
# Scheme 4 — Fixed-length ArcMark (non-adaptive baseline)
# ============================================================================
# Ported from compare.py (fixed_length_trial / get_code), trimmed to the two
# metrics this ablation keeps: message correctness and token count. The payload
# is embedded as a length-n random-linear codeword driven through the SAME
# ArcMark OT channel and keying (SHARED_SEED, ARC_CONFIG, P_FIELD, R_RESOLUTION,
# PHI) as the posterior-matching schemes above, then decoded by correlating the
# emitted tokens against the codebook — no belief update, no stopping rule.
def get_code(n_tokens: int, n_msg: int) -> RandomLinearCode:
    """Cache one random-linear code per (codeword_length, num_messages)."""
    key = (n_tokens, n_msg)
    code = CTX.code_cache.get(key)
    if code is None:
        code = RandomLinearCode.build(
            num_messages=n_msg,
            codeword_length=n_tokens,
            alphabet_size=P_FIELD,
            seed=CODE_SEED,
        )
        CTX.code_cache[key] = code
    return code


@torch.no_grad()
def arcmark_fixed_length_trial(prompt_ids, payload_int, n_tokens, n_msg):
    """Non-adaptive fixed-length ArcMark. Returns (ok, n_used).

    Unlike bam_fixed_length (Scheme 3), the codeword — and therefore the token
    trajectory — depends on n_tokens, so each n needs its own generation.
    """
    model = CTX.model
    tokenizer = CTX.tokenizer
    vocab_size = CTX.vocab_size
    code = get_code(n_tokens, n_msg)
    codeword = code.encode(payload_int)

    proc = ArcMarkLogitsProcessor(
        vocab_size=vocab_size,
        alphabet_size=P_FIELD,
        num_keys=R_RESOLUTION,
        seed=SHARED_SEED,
        phi=PHI,
        temperature=1.0,
        config=ARC_CONFIG,
        side_info_mode=SIDE_INFO_MODE,
        tokenizer=tokenizer,
    )
    prompt_len = len(prompt_ids)
    proc.set_trial(codeword=codeword, prompt_length=prompt_len)

    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    gen_kwargs = ArcMarkLogitsProcessor.default_generate_kwargs(n_tokens)
    gen_kwargs["min_new_tokens"] = n_tokens
    gen_kwargs["eos_token_id"] = None
    output = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=[proc],
        **gen_kwargs,
    )
    gen_tokens = output[0, prompt_len:]
    n_got = int(gen_tokens.shape[0])

    if n_got < n_tokens:
        # Rare short generation: decode against the truncated codebook.
        result = decode_message(
            gen_tokens.cpu(),
            vocab_size=vocab_size,
            alphabet_size=P_FIELD,
            num_keys=R_RESOLUTION,
            seed=SHARED_SEED,
            codewords=code.codebook[:, :n_got],
            phi=PHI,
            scoring="log",
            config=ARC_CONFIG,
        )
        return int(result.message_idx) == payload_int, n_got

    result = decode_with_code(
        gen_tokens.cpu(),
        vocab_size=vocab_size,
        num_keys=R_RESOLUTION,
        seed=SHARED_SEED,
        code=code,
        phi=PHI,
        scoring="log",
        config=ARC_CONFIG,
    )
    return int(result.message_idx) == payload_int, n_tokens


# ============================================================================
# Summary helper — mean + SE for error and length
# ============================================================================
def summarize(rs):
    """rs: list of (ok, n_tokens, why). Returns
    (err, err_se, tok, tok_std, tok_se, forced_frac)."""
    n = len(rs)
    errs = [0.0 if r[0] else 1.0 for r in rs]
    p = float(np.mean(errs))
    err_se = math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")
    tok_vals = [r[1] for r in rs]
    tok = float(np.mean(tok_vals))
    std = float(np.std(tok_vals))
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    forced = float(np.mean([1.0 if r[2] == "forced" else 0.0 for r in rs]))
    return p, err_se, tok, std, sem, forced


# ============================================================================
# Per-model sweep
# ============================================================================
def run_model(model_name: str) -> list[dict]:
    global CTX
    CTX = LMContext(model_name)
    rows: list[dict] = []
    try:
        CTX.prompt_pool = build_prompt_pool()
        pool = CTX.prompt_pool

        for k_bits in K_BITS_LIST:
            n_msg = 2 ** k_bits

            # ── Variable-length schemes: sweep L ────────────────────────────
            for L in L_VALUES:
                thresh = 1.0 - 1.0 / L
                for scheme_kind in ("BAM", "1PH"):
                    name = f"{scheme_kind}-K{k_bits}-L{L}"
                    log("\n" + "=" * 72)
                    if scheme_kind == "BAM":
                        log(f"[{model_name}] {name}  (two-phase: g1={GAMMA} "
                            f"rACK={thresh:.6f} rNACK={RHO_NACK})  "
                            f"M={n_msg}  — {N_TRIALS} trials")
                    else:
                        log(f"[{model_name}] {name}  (one-phase: commit at "
                            f"pi_max >= {thresh:.6f}, no confirmation)  "
                            f"M={n_msg}  — {N_TRIALS} trials")
                    log("=" * 72)
                    rs = []
                    for i in range(N_TRIALS):
                        prompt_ids = pool[i % len(pool)]
                        # Paired randomness: identical (prompt, message, seed)
                        # for all schemes and every L at fixed (k_bits, i).
                        trial_seed = 50000 + 1000000 * k_bits + i
                        m_true = int(np.random.RandomState(trial_seed)
                                     .randint(n_msg))
                        np.random.seed(trial_seed)
                        torch.manual_seed(trial_seed)
                        t0 = time.time()
                        if scheme_kind == "BAM":
                            ok, n, why = bam_two_phase(
                                prompt_ids, m_true, n_msg, MAX_TOKENS,
                                GAMMA, thresh, RHO_NACK)
                        else:
                            ok, n, why = one_phase_threshold(
                                prompt_ids, m_true, n_msg, MAX_TOKENS, thresh)
                        dt = time.time() - t0
                        rs.append((ok, n, why))
                        log(f"  trial {i + 1:>4}/{N_TRIALS}: m={m_true:>5} -> "
                            f"{'OK' if ok else 'WRONG':>5} n={n:>3} ({why}) "
                            f"[{dt:.1f}s]")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    err, err_se, tok, tok_std, tok_se, forced = summarize(rs)
                    log(f"  --> err={err:.4f}±{err_se:.4f}(SE)  "
                        f"avg_tok={tok:.2f}±{tok_se:.2f}(SE)  "
                        f"std_tok={tok_std:.2f}  forced_frac={forced:.3f}")
                    rows.append({
                        "model": model_name, "scheme": name,
                        "kind": scheme_kind, "k_bits": k_bits, "L": L,
                        "err": err, "err_se": err_se,
                        "tok": tok, "tok_std": tok_std, "tok_se": tok_se,
                        "forced_frac": forced,
                    })

            # ── Scheme 3: BAM-fixed-length — sweep n, one rollout/trial ─────
            log("\n" + "=" * 72)
            log(f"[{model_name}] FIX-K{k_bits}  (fixed-length: commit to "
                f"argmax at n in {FIXED_LENGTHS}, no threshold)  "
                f"M={n_msg}  — {N_TRIALS} trials")
            log("=" * 72)
            per_n: dict[int, list] = {n_fix: [] for n_fix in FIXED_LENGTHS}
            for i in range(N_TRIALS):
                prompt_ids = pool[i % len(pool)]
                # Same paired randomness as the variable-length schemes.
                trial_seed = 50000 + 1000000 * k_bits + i
                m_true = int(np.random.RandomState(trial_seed)
                             .randint(n_msg))
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                t0 = time.time()
                res = bam_fixed_length(prompt_ids, m_true, n_msg,
                                       FIXED_LENGTHS)
                dt = time.time() - t0
                for n_fix, ok in res.items():
                    per_n[n_fix].append((ok, n_fix, "fixed"))
                summary = " ".join(
                    f"n{n_fix}:{'OK' if res[n_fix] else 'WR'}"
                    for n_fix in sorted(res))
                log(f"  trial {i + 1:>4}/{N_TRIALS}: m={m_true:>5} -> "
                    f"{summary} [{dt:.1f}s]")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for n_fix in FIXED_LENGTHS:
                rs = per_n[n_fix]
                err, err_se, tok, tok_std, tok_se, forced = summarize(rs)
                log(f"  --> n={n_fix:>3}: err={err:.4f}±{err_se:.4f}(SE)  "
                    f"tok={tok:.0f}")
                rows.append({
                    "model": model_name,
                    "scheme": f"FIX-K{k_bits}-n{n_fix}",
                    "kind": "FIX", "k_bits": k_bits, "L": n_fix,
                    "err": err, "err_se": err_se,
                    "tok": tok, "tok_std": tok_std, "tok_se": tok_se,
                    "forced_frac": forced,
                })

            # ── Scheme 4: Fixed-length ArcMark — non-adaptive baseline ──────
            # One generation PER n (the codeword, hence the token trajectory,
            # depends on n — unlike Scheme 3, which reuses a single rollout).
            log("\n" + "=" * 72)
            log(f"[{model_name}] FL-K{k_bits}  (fixed-length ArcMark: length-n "
                f"codeword, correlation decode, n in {FL_LENGTHS})  "
                f"M={n_msg}  — {N_TRIALS} trials")
            log("=" * 72)
            fl_per_n: dict[int, list] = {n_fix: [] for n_fix in FL_LENGTHS}
            for i in range(N_TRIALS):
                prompt_ids = pool[i % len(pool)]
                # Same paired randomness as the other three schemes: identical
                # (prompt, message) for a given (k_bits, i).
                trial_seed = 50000 + 1000000 * k_bits + i
                m_true = int(np.random.RandomState(trial_seed)
                             .randint(n_msg))
                t0 = time.time()
                per_n_ok: dict[int, bool] = {}
                for n_fix in FL_LENGTHS:
                    # Reseed per n so each (trial, n) generation is reproducible
                    # while the message stays fixed across schemes for trial i.
                    np.random.seed(trial_seed)
                    torch.manual_seed(trial_seed)
                    ok, n_used = arcmark_fixed_length_trial(
                        prompt_ids, m_true, n_fix, n_msg)
                    per_n_ok[n_fix] = ok
                    fl_per_n[n_fix].append((ok, n_used, "fixed"))
                dt = time.time() - t0
                summary = " ".join(
                    f"n{n_fix}:{'OK' if per_n_ok[n_fix] else 'WR'}"
                    for n_fix in sorted(per_n_ok))
                log(f"  trial {i + 1:>4}/{N_TRIALS}: m={m_true:>5} -> "
                    f"{summary} [{dt:.1f}s]")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for n_fix in FL_LENGTHS:
                rs = fl_per_n[n_fix]
                err, err_se, tok, tok_std, tok_se, forced = summarize(rs)
                log(f"  --> n={n_fix:>3}: err={err:.4f}±{err_se:.4f}(SE)  "
                    f"tok={tok:.0f}")
                rows.append({
                    "model": model_name,
                    "scheme": f"FL-K{k_bits}-n{n_fix}",
                    "kind": "FL", "k_bits": k_bits, "L": n_fix,
                    "err": err, "err_se": err_se,
                    "tok": tok, "tok_std": tok_std, "tok_se": tok_se,
                    "forced_frac": forced,
                })
    finally:
        CTX.teardown()
        CTX = None
    return rows


# ============================================================================
# Run all models
# ============================================================================
all_results: list[dict] = []

for model_name in MODEL_NAMES:
    log("\n" + "#" * 72)
    log(f"# MODEL: {model_name}")
    log("#" * 72)
    try:
        all_results.extend(run_model(model_name))
    except Exception as e:
        log(f"!! Model {model_name} failed: {e!r}")
        if CTX is not None:
            try:
                CTX.teardown()
            except Exception:
                pass
            CTX = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Combined report + CSV
# ============================================================================
log("\n" + "=" * 72); log("COMBINED SUMMARY (all models)"); log("=" * 72)
log(f"\n{'Model':<24} {'Scheme':<18} {'K':>3} {'L/n':>7} "
    f"{'err':>8} {'err_se':>8} {'avg_tok':>8} {'tok_se':>7} {'forced':>7}")
log("-" * 100)
last_model = None
for row in all_results:
    short = row["model"].split('/')[-1]
    if short != last_model:
        if last_model is not None:
            log("-" * 100)
        last_model = short
    log(f"{short:<24} {row['scheme']:<18} {row['k_bits']:>3} {row['L']:>7} "
        f"{row['err']:>8.4f} {row['err_se']:>8.4f} "
        f"{row['tok']:>8.2f} {row['tok_se']:>7.2f} {row['forced_frac']:>7.3f}")

with open(OUT_CSV, "w") as f:
    # For kind=FIX rows the "L_or_n" column holds the fixed token length n;
    # for BAM/1PH rows it holds the swept L.
    f.write("model,scheme,kind,k_bits,L_or_n,"
            "err_rate,err_se,avg_tokens,std_tokens,se_tokens,forced_frac\n")
    for row in all_results:
        f.write(f"{row['model']},{row['scheme']},{row['kind']},"
                f"{row['k_bits']},{row['L']},"
                f"{row['err']:.6f},{row['err_se']:.6f},"
                f"{row['tok']:.4f},{row['tok_std']:.4f},{row['tok_se']:.4f},"
                f"{row['forced_frac']:.4f}\n")
log(f"\nWrote {OUT_CSV}")


# ============================================================================
# Frontier plot — 4 curves per model (8-bit), stacked linear/semilog panels.
# Merged from plot_ablation.py. The 4th curve (fixed-length ArcMark, kind="FL")
# ============================================================================
OUT_FRONTIER_PNG = "ablation_confirmation_frontier.png"
OUT_FRONTIER_PDF = "ablation_confirmation_frontier.pdf"

N_DROP = 0              # drop the last (highest-L) N_DROP points of each curve

GOLDEN = (1 + 5 ** 0.5) / 2          # ~1.618, per-panel width:height ratio
PANEL_W = 6.8                        # panel width in inches

# One color per SCHEME. Variable-length: solid lines, filled markers.
# Fixed-length: dashed lines, open markers.
# kind -> (color, marker, linestyle, fillstyle, label)
CURVE_STYLE = {
    "BAM": ("#d62728", "s", "-",  "full", "BAM"),
    "1PH": ("#1f77b4", "o", "-",  "full", "BAM-One-phase"),
    "FIX": ("#2ca02c", "D", "--", "none", "BAM-fixed-length"),
    "FL":  ("#9467bd", "^", "--", "none", "ArcMark"),
}
FIXED_KINDS = {"FIX", "FL"}          # deterministic length: no horizontal bars

ANNOTATE_L = {2, 16, 128, 2048, 32768}    # variable-length curves (L values)
ANNOTATE_N = {20, 40, 60, 80}             # fixed-length curves (n values)

# ── Gap illustration (semilog panel only) ───────────────────────────────────
#   ArcMark -> BAM-fixed-length : rate gain from posterior matching
#   BAM-fixed-length -> One-phase : sequentiality gain
#   One-phase -> BAM : adaptivity gain
GAP_PAIRS = [
    # (kind_top, kind_bottom, label, label_placement, x_tokens)
    ("FL",  "FIX", "posterior-matching rate gain", "top",       30.0),
    ("FIX", "1PH", "sequentiality gain",           "right_low", 38.0),
    ("1PH", "BAM", "adaptivity gain",              "right_mid", 38.0),
]
GAP_COLOR = "0.25"
GAP_MIN_ARROW = 0.07  # min gap height (axes fraction) to draw the <-> arrow


def _err_at_x(cur: pd.DataFrame, x0: float):
    """Error rate of a curve at token count x0, interpolating linearly in
    (tokens, log error). None if x0 is outside the curve's token range."""
    cur = cur.sort_values("avg_tokens")
    xs = cur["avg_tokens"].to_numpy(dtype=float)
    ys = cur["err_rate"].to_numpy(dtype=float)
    if len(xs) < 2 or x0 < xs[0] or x0 > xs[-1]:
        return None
    i = int(min(max((xs <= x0).sum() - 1, 0), len(xs) - 2))
    f = (x0 - xs[i]) / (xs[i + 1] - xs[i])
    return math.exp(math.log(ys[i]) + f * (math.log(ys[i + 1]) - math.log(ys[i])))


df = pd.read_csv(OUT_CSV)
df = df[df["k_bits"] == 8]

models = list(df["model"].unique())
n_models = len(models)

# Stacked layout: for each model, (a) linear on top, (b) semilog below.
panel_h = PANEL_W / GOLDEN
fig, axes = plt.subplots(2 * max(1, n_models), 1,
                         figsize=(PANEL_W, 2 * max(1, n_models) * panel_h),
                         squeeze=False)
axes = axes[:, 0]

for mi, model in enumerate(models):
    sub = df[df["model"] == model]
    panels = ((axes[2 * mi],     "linear", "(a)"),
              (axes[2 * mi + 1], "log",    "(b)"))
    for ax, yscale, tag in panels:
        for kind, (color, marker, ls, fs, label) in CURVE_STYLE.items():
            cur = sub[sub["kind"] == kind].sort_values("L_or_n")
            if N_DROP > 0 and kind not in FIXED_KINDS:
                cur = cur.iloc[:-N_DROP]
            if cur.empty:
                continue
            xerr = None if kind in FIXED_KINDS else cur["se_tokens"]
            yerr = cur["err_se"] if yscale == "log" else None
            ax.errorbar(
                cur["avg_tokens"], cur["err_rate"],
                xerr=xerr, yerr=yerr,
                marker=marker, linestyle=ls, fillstyle=fs, color=color,
                markersize=6.5, linewidth=1.6, capsize=2.5, elinewidth=1.0,
                label=label,
            )
            annotate = ANNOTATE_N if kind in FIXED_KINDS else ANNOTATE_L
            fmt = "n={}" if kind in FIXED_KINDS else "L={}"
            for _, r in cur.iterrows():
                if int(r["L_or_n"]) in annotate:
                    ax.annotate(fmt.format(int(r["L_or_n"])),
                                (r["avg_tokens"], r["err_rate"]),
                                textcoords="offset points", xytext=(6, 5),
                                fontsize=7, color=color)
        ax.set_xlabel("Average tokens", fontsize=13)
        ax.set_ylabel("Message error rate", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11, loc="upper right")
        if yscale == "log":
            from matplotlib.patches import Ellipse

            def to_axes(x, y, _ax=ax):
                return tuple(_ax.transAxes.inverted()
                             .transform(_ax.transData.transform((x, y))))

            for k_top, k_bot, gap_label, place, gx in GAP_PAIRS:
                y_top = _err_at_x(sub[sub["kind"] == k_top], gx)
                y_bot = _err_at_x(sub[sub["kind"] == k_bot], gx)
                if y_top is None or y_bot is None:
                    continue
                cx, ay_top = to_axes(gx, y_top)
                _,  ay_bot = to_axes(gx, y_bot)
                gap_h = abs(ay_top - ay_bot)
                ell = Ellipse((cx, 0.5 * (ay_top + ay_bot)),
                              width=0.055, height=gap_h + 0.05,
                              transform=ax.transAxes, fill=False,
                              edgecolor=GAP_COLOR, linestyle=":", lw=1.3)
                ax.add_patch(ell)
                if gap_h >= GAP_MIN_ARROW:
                    ax.annotate("", xy=(gx, y_bot), xytext=(gx, y_top),
                                arrowprops=dict(arrowstyle="<->",
                                                color=GAP_COLOR,
                                                lw=1.5, shrinkA=2, shrinkB=2))
                box = dict(facecolor="white", alpha=0.85,
                           edgecolor="none", pad=1.5)
                if place == "top":
                    ax.text(gx + 2.6, y_top * 1.35, gap_label,
                            ha="left", va="center", fontsize=9.5,
                            color=GAP_COLOR, bbox=box)
                elif place == "right_low":
                    ax.text(gx + 3.2, y_bot * 1.35, gap_label,
                            ha="left", va="center", fontsize=9.5,
                            color=GAP_COLOR, bbox=box)
                else:  # "right_mid"
                    ax.text(gx + 3.2, math.sqrt(y_top * y_bot), gap_label,
                            ha="left", va="center", fontsize=9.5,
                            color=GAP_COLOR, bbox=box)
        ax.text(0.01, 1.02, tag, transform=ax.transAxes,
                fontsize=11, va="bottom")

plt.tight_layout()
plt.savefig(OUT_FRONTIER_PNG, dpi=160)
plt.savefig(OUT_FRONTIER_PDF)
log(f"Wrote {OUT_FRONTIER_PNG} and {OUT_FRONTIER_PDF}\nDone.")