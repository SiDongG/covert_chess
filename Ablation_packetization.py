"""
ablation_packetization.py — Packetization ablation on original (two-phase) BAM,
with a per-component runtime breakdown.

A fixed 16-bit payload is embedded with original BAM in three ways:

  A. "1x24": one packet of 24 bits    -> posterior over M = 2^24 = 16,777,216
  B. "2x12": two packets of 12 bits   -> two sequential BAM runs, M = 4096 each
  C. "3x8":  three packets of 8 bits  -> three sequential BAM runs, M = 256 each

Packets run back-to-back on ONE continuous LM stream under a single global
MAX_TOKENS budget; a trial is correct iff EVERY packet decodes correctly.

RUNTIME BREAKDOWN (per emitted token, wall-clock, CUDA-synchronized at every
boundary so async GPU work is attributed to the right component):

  t_enc  ENCODING: message -> sampling rule. The CDF symbol map (trivial)
         plus side-info/permutation derivation and the OT solve
         (solve_arcmark_ot + extract_conditional). EXCLUDES the posterior
         computation: in practice the posterior is maintained at BOTH encoder
         and decoder, so charging it here as well would double-count it and
         mask the OT cost we want to isolate.
  t_gen  TOKEN GENERATION: the LLM itself — softmax of the cached logits,
         the multinomial draw from the OT conditional, and the incremental
         forward pass (KV-cache advance).
  t_dec  DECODING: token -> angle, the per-symbol likelihood, the O(M)
         posterior update (message_likelihood + Bayes step + fresh CDF), and
         the threshold check (trivial).

Efficiency changes vs the previous version:
  * side-info + permutation are derived ONCE per step and shared between
    emission and angle read-out (previously computed twice per token);
  * the posterior CDF is computed once per step and reused by both the
    encoder's symbol map and the decoder's likelihood (previously two
    cumsums per token — this halves the O(M) posterior-side cost);
  * the knockdown multiply is skipped while knockdown is identity (i.e.
    outside retry episodes, which is almost always);
  * torch.inference_mode for all LM calls;
  * gc / cuda.empty_cache every GC_EVERY trials instead of every trial.
  NOT done: batching multiple trials through the LM in one forward. It would
  give the largest throughput win but shares one forward across trials with
  different stopping times, which destroys the per-component / per-token
  attribution this script exists to measure.

L is sieved to a few representative operating points (edit L_VALUES to
change). Trials are PAIRED across the three schemes and all L values.
"""

from __future__ import annotations

import math
import os
import sys
import time
import gc

import numpy as np
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


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAMES = [
    "unsloth/Meta-Llama-3.1-8B",
    #"unsloth/Qwen3.5-9B-Base",
    #"unsloth/mistral-7b-v0.3",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False
N_TRIALS   = 1000 if not SMOKE_TEST else 4
GC_EVERY   = 25          # run gc + cuda.empty_cache every this many trials

K_BITS_TOTAL = 24    # total payload; must be divisible by every packet count

# Packetization schemes: (name, n_packets). Packet size = K_BITS_TOTAL / n.
PACKET_SCHEMES = [
    ("1x24", 1),     # A: one 24-bit packet,     M = 16,777,216
    ("2x12", 2),     # B: two  12-bit packets,   M = 4096 each
    ("3x8",  3),     # C: three 8-bit packets,   M = 256 each
]

# Posteriors with M >= this many messages are kept as float64 torch tensors on
# the model's device (GPU) instead of numpy: for M = 2^24 the O(M) cumsum /
# likelihood / Bayes step per token takes seconds on CPU but ~ms on GPU
# (memory-bandwidth bound). float64 is used for safety: the CDF must stay
# accurate after ~100 multiplicative Bayes updates on a heavily skewed
# posterior, and float64 makes the cumsum/normalization error a non-issue
# rather than something to audit. Adds ~0.8 GB of GPU memory at M=2^24. The
# decoding component still honestly reflects the O(M) cost — it just measures
# the cost of the sensible (parallel) implementation. (On a CPU-only machine
# the 1x24 scheme is impractical either way; run this on GPU.)
TORCH_POSTERIOR_MIN_M = 1 << 14

# Sieved, representative rho_ACK = 1 - 1/L operating points spanning the
# frontier (low / mid / high / very-high reliability targets).
L_VALUES = [4, 64, 2048, 32768] if not SMOKE_TEST else [8]

# ── BAM fixed parameters (as in compare.py) ─────────────────────────────────
GAMMA      = 0.5     # communication-phase decision threshold g1
RHO_NACK   = 0.75    # rho_NACK
EPS_NOISE  = 0.4     # eps      (communication-phase Laplace floor)
EPS_CONF   = 0.4     # eps_ACK  (confirmation-phase antipodal floor)

MAX_TOKENS      = 1000   # GLOBAL budget shared by all packets of a trial
MAX_CONF_STEPS  = 120
MIN_COMM_TOKENS = 0

OUT_PLOT = "ablation_packetization.png"
OUT_CSV  = "ablation_packetization.csv"

# ── Shared ArcMark core knobs ───────────────────────────────────────────────
P_FIELD            = 4
R_RESOLUTION       = 4
SHARED_SEED        = 0xA12C
TOP_K              = 50
SINKHORN_REG       = 0.2
SINKHORN_MAX_ITER  = 4000
SINKHORN_STOP_THR  = 1e-4
PHI                = 0.0

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


def _sync():
    """Barrier so async CUDA work lands inside the right timing block."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


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
        self.prompt_pool: list[list[int]] = []
        log(f"Loaded {model_name}. vocab_size={self.vocab_size}")

    def teardown(self):
        self.model = None
        self.tokenizer = None
        self.perm_cache.clear()
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
        with torch.inference_mode():
            out = model(ids, use_cache=True)
        self.past = out.past_key_values
        self._last_logits = out.logits[0, -1].float()

    @torch.inference_mode()
    def probs(self) -> torch.Tensor:
        return torch.softmax(self._last_logits, dim=-1)

    @torch.inference_mode()
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
# Shared primitives — side info derived ONCE per step
# ============================================================================
def _context_tokens_for_step(emitted: list[int], context_width: int) -> tuple[int, ...]:
    pad_len = max(0, context_width - len(emitted))
    return tuple([0] * pad_len + emitted[-context_width:])


def step_side_info(emitted: list[int]):
    """Derive (s_index, perm) for the CURRENT step once; reused by both the
    OT emission and the angle read-out."""
    context_tokens = _context_tokens_for_step(emitted, ARC_CONFIG.context_width)
    s_index, perm_seed = compute_key_si(
        secret_key=SHARED_SEED,
        context_tokens=context_tokens,
        num_keys=R_RESOLUTION,
        mode=SIDE_INFO_MODE,
        tokenizer=CTX.tokenizer,
    )
    perm = _perm_for_seed(perm_seed, CTX.model.device)
    return s_index, perm


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


@torch.inference_mode()
def build_sampling_rule(probs: torch.Tensor, symbol: int,
                        s_index: int, perm: torch.Tensor) -> torch.Tensor:
    """ENCODING core: OT solve + conditional extraction -> the sampling rule
    (a distribution over tokens). Posterior computation is NOT part of this."""
    ot_result = solve_arcmark_ot(
        probs,
        codeword_symbol=int(symbol),
        alphabet_size=P_FIELD,
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
    return cond


def angle_from_token(token_id: int, s_index: int, perm: torch.Tensor) -> float:
    permuted_id = int(perm[token_id].item())
    theta = (2.0 * math.pi) * permuted_id / float(CTX.vocab_size)
    s_angle = (2.0 * math.pi) * s_index / float(R_RESOLUTION)
    return (theta - s_angle) % (2.0 * math.pi)


# ============================================================================
# Communication-phase likelihood (LAPLACE core, full 4-symbol)
# ============================================================================
P_SYM = P_FIELD

_SIGMA     = math.pi / P_SYM
_B_LAPLACE = float(os.environ.get("LAPLACE_B", _SIGMA / math.sqrt(2.0)))
_Z_SIGNAL = 2.0 * _B_LAPLACE * (1.0 - math.exp(-math.pi / _B_LAPLACE))


def symbol_from_cdf(cdf, pi, m: int, R: float) -> int:
    """Posterior-matching CDF symbol map, reusing the step's precomputed CDF
    (cdf[j] = sum(pi[:j]), len M+1). Works on numpy arrays and torch tensors."""
    V = float(cdf[m]) + R * float(pi[m])
    return min(int(P_SYM * V), P_SYM - 1)


def per_symbol_likelihood(angle_obs: float) -> np.ndarray:
    signal = np.empty(P_SYM)
    for u in range(P_SYM):
        target = (2 * math.pi * u / P_SYM + PHI) % (2 * math.pi)
        d = abs(angle_obs - target) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        signal[u] = math.exp(-d / _B_LAPLACE) / _Z_SIGNAL
    ells = (1.0 - EPS_NOISE) * signal + EPS_NOISE / (2.0 * math.pi)
    return ells


def message_likelihood_from_cdf(ells: np.ndarray, pi, cdf):
    """Vectorized message likelihood reusing the step's precomputed CDF.
    Identical to the original per-message loop (verified to 1e-12). Dual
    backend: numpy for small M, torch (float64, on the posterior's device)
    for large M."""
    lo = cdf[:-1]
    hi = cdf[1:]
    if torch.is_tensor(pi):
        width = (hi - lo).clamp_min(1e-30)
        q = torch.zeros_like(pi)
        for u in range(P_SYM):
            ul, uh = u / P_SYM, (u + 1) / P_SYM
            ov = (hi.clamp(max=uh) - lo.clamp(min=ul)).clamp_min(0.0)
            q += (ov / width) * float(ells[u])
        return q
    width = np.maximum(hi - lo, 1e-30)
    q = np.zeros(len(pi))
    for u in range(P_SYM):
        ul, uh = u / P_SYM, (u + 1) / P_SYM
        ov = np.maximum(0.0, np.minimum(hi, uh) - np.maximum(lo, ul))
        q += (ov / width) * ells[u]
    return q


def make_cdf(pi):
    if torch.is_tensor(pi):
        cdf = torch.zeros(len(pi) + 1, dtype=pi.dtype, device=pi.device)
        torch.cumsum(pi, dim=0, out=cdf[1:])
        return cdf
    return np.concatenate([[0.0], np.cumsum(pi)])


def uniform_posterior(n_msg: int):
    """Uniform prior over n_msg messages, on the appropriate backend."""
    if n_msg >= TORCH_POSTERIOR_MIN_M and DEVICE == "cuda":
        return torch.full((n_msg,), 1.0 / n_msg, dtype=torch.float64,
                          device=CTX.model.device)
    return np.ones(n_msg) / n_msg


def ones_like_posterior(pi):
    return torch.ones_like(pi) if torch.is_tensor(pi) else np.ones_like(pi)


# ============================================================================
# Timing accumulator
# ============================================================================
class StepTimes:
    __slots__ = ("enc", "gen", "dec", "n")

    def __init__(self):
        self.enc = 0.0
        self.gen = 0.0
        self.dec = 0.0
        self.n = 0

    def per_token_ms(self) -> tuple[float, float, float]:
        if self.n == 0:
            return (float("nan"),) * 3
        return (1000.0 * self.enc / self.n,
                1000.0 * self.gen / self.n,
                1000.0 * self.dec / self.n)


# ============================================================================
# One timed COMM step (posterior-matching over M messages)
# ============================================================================
def run_comm_step(pi, cdf, m_true, emitted, lm: "IncrementalLM",
                  tt: StepTimes):
    """Returns (pi_new, cdf_new). Timing attribution:
      gen: probs softmax | multinomial draw + KV-cache advance
      enc: CDF symbol map + side-info/perm + OT solve + conditional extract
      dec: angle read-out + likelihoods + posterior update + fresh CDF
    """
    _sync(); t0 = time.perf_counter()
    probs = lm.probs()
    _sync(); t1 = time.perf_counter()

    R = float(np.random.random())
    u = symbol_from_cdf(cdf, pi, m_true, R)
    s_index, perm = step_side_info(emitted)
    cond = build_sampling_rule(probs, u, s_index, perm)
    _sync(); t2 = time.perf_counter()

    x = int(torch.multinomial(cond, num_samples=1).item())
    emitted.append(x)
    lm.advance(x)
    _sync(); t3 = time.perf_counter()

    angle = angle_from_token(x, s_index, perm)
    ells = per_symbol_likelihood(angle)
    q = message_likelihood_from_cdf(ells, pi, cdf)
    pi_new = pi * q
    s = float(pi_new.sum())
    if s > 0:
        pi_new = pi_new / s
    else:
        pi_new = uniform_posterior(len(pi))
    cdf_new = make_cdf(pi_new)
    _sync()          # decode may run on GPU (large-M posterior)
    t4 = time.perf_counter()

    tt.gen += (t1 - t0) + (t3 - t2)
    tt.enc += (t2 - t1)
    tt.dec += (t4 - t3)
    tt.n += 1
    return pi_new, cdf_new


# ============================================================================
# Confirmation phase — 2-symbol ANTIPODAL channel, timed the same way
# ============================================================================
SYM_ACK  = 0                                     # angle 0
SYM_NACK = 2                                     # angle pi (with P_FIELD=4)
_ANGLE_ACK  = (2 * math.pi * SYM_ACK  / P_SYM + PHI) % (2 * math.pi)
_ANGLE_NACK = (2 * math.pi * SYM_NACK / P_SYM + PHI) % (2 * math.pi)
_B_CONF  = _B_LAPLACE
_Z_CONF  = 2.0 * _B_CONF * (1.0 - math.exp(-math.pi / _B_CONF))


def _circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def conf_two_symbol_likelihood(angle_obs: float) -> np.ndarray:
    d_ack  = _circ_dist(angle_obs, _ANGLE_ACK)
    d_nack = _circ_dist(angle_obs, _ANGLE_NACK)
    sig_ack  = math.exp(-d_ack  / _B_CONF) / _Z_CONF
    sig_nack = math.exp(-d_nack / _B_CONF) / _Z_CONF
    ell_ack  = (1.0 - EPS_CONF) * sig_ack  + EPS_CONF / (2.0 * math.pi)
    ell_nack = (1.0 - EPS_CONF) * sig_nack + EPS_CONF / (2.0 * math.pi)
    return np.array([ell_ack, ell_nack])


def run_confirmation(true_bit, emitted, lm: "IncrementalLM", max_steps,
                     g_ack, g_nack, tt: StepTimes):
    tx_symbol = SYM_ACK if true_bit == 0 else SYM_NACK
    rho = np.array([0.5, 0.5])
    used = 0
    for _ in range(max_steps):
        _sync(); t0 = time.perf_counter()
        probs = lm.probs()
        _sync(); t1 = time.perf_counter()

        s_index, perm = step_side_info(emitted)
        cond = build_sampling_rule(probs, tx_symbol, s_index, perm)
        _sync(); t2 = time.perf_counter()

        x = int(torch.multinomial(cond, num_samples=1).item())
        emitted.append(x)
        lm.advance(x)
        _sync(); t3 = time.perf_counter()

        angle = angle_from_token(x, s_index, perm)
        ell = conf_two_symbol_likelihood(angle)
        rho = rho * ell
        rho /= rho.sum()
        t4 = time.perf_counter()

        tt.gen += (t1 - t0) + (t3 - t2)
        tt.enc += (t2 - t1)
        tt.dec += (t4 - t3)
        tt.n += 1
        used += 1
        if rho[0] >= g_ack:  return "ACK",  used
        if rho[1] >= g_nack: return "NACK", used
    return ("ACK" if rho[0] >= rho[1] else "NACK"), used


# ============================================================================
# One BAM packet on an EXISTING stream (shared lm/emitted, shared budget)
# ============================================================================
def bam_packet(m_true_pkt: int, n_msg: int, budget: int,
               lm: "IncrementalLM", emitted: list[int],
               g1: float, ra: float, rn: float,
               tt: StepTimes,
               min_comm: int = MIN_COMM_TOKENS):
    """Run one two-phase BAM packet, continuing on the given LM stream.

    Returns (decoded, tokens_used, why). The knockdown multiply is skipped
    while knockdown is identity (kd_active False), which is the common case.
    """
    pi = uniform_posterior(n_msg)
    cdf = make_cdf(pi)
    knockdown = ones_like_posterior(pi)
    kd_active = False
    t = 0
    while t < budget:
        pi, cdf = run_comm_step(pi, cdf, m_true_pkt, emitted, lm, tt)
        t += 1
        if t < min_comm:
            continue
        # Threshold check is O(M) (max over the posterior) — for large M it is
        # non-trivial, so it is timed and charged to DECODING (no n increment:
        # it belongs to the token just emitted).
        _sync(); td0 = time.perf_counter()
        if kd_active:
            eff = pi * knockdown
            eff = eff / eff.sum()
        else:
            eff = pi
        crossed = float(eff.max()) >= g1
        cand = int(eff.argmax()) if crossed else -1
        _sync(); tt.dec += time.perf_counter() - td0
        if crossed:
            true_bit = 0 if cand == m_true_pkt else 1
            conf_budget = min(MAX_CONF_STEPS, budget - t)
            if conf_budget <= 0:
                break
            outcome, ct = run_confirmation(
                true_bit, emitted, lm, conf_budget, ra, rn, tt)
            t += ct
            if outcome == "ACK":
                return cand, t, "ack"
            knockdown[cand] *= (1 - rn) / rn
            pi = pi * knockdown
            pi = pi / pi.sum()
            cdf = make_cdf(pi)
            knockdown = ones_like_posterior(pi)
            kd_active = False
            if t >= budget:
                break
    if kd_active:
        eff = pi * knockdown
        eff = eff / eff.sum()
    else:
        eff = pi
    return int(eff.argmax()), t, "forced"


def split_message(m: int, n_packets: int, k_total: int = K_BITS_TOTAL) -> list[int]:
    """Split a k_total-bit message into n_packets equal-size sub-messages,
    most-significant packet first."""
    k_pkt = k_total // n_packets
    mask = (1 << k_pkt) - 1
    return [(m >> (k_pkt * (n_packets - 1 - j))) & mask
            for j in range(n_packets)]


def bam_packetized(prompt_ids, m_true: int, n_packets: int,
                   g1: float, ra: float, rn: float):
    """Transmit the 16-bit message as n_packets sequential BAM packets on one
    continuous LM stream with a single global MAX_TOKENS budget.

    Returns (ok, total_tokens, why, tt) where tt is the trial's StepTimes.
    """
    k_pkt = K_BITS_TOTAL // n_packets
    n_msg = 2 ** k_pkt
    sub_msgs = split_message(m_true, n_packets)

    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    tt = StepTimes()
    total = 0
    decoded_parts: list[int] = []
    any_forced = False
    try:
        for j in range(n_packets):
            budget = MAX_TOKENS - total
            if budget <= 0:
                decoded_parts.append(0)   # out of budget: uniform-prior argmax
                any_forced = True
                continue
            dec, used, why = bam_packet(
                sub_msgs[j], n_msg, budget, lm, emitted, g1, ra, rn, tt)
            decoded_parts.append(dec)
            total += used
            if why == "forced":
                any_forced = True
        ok = all(d == s for d, s in zip(decoded_parts, sub_msgs))
        return ok, total, ("forced" if any_forced else "ack"), tt
    finally:
        lm.free()


# ============================================================================
# Summary helper — mean + SE for error, length, and per-component times
# ============================================================================
def _mss(vals: list[float]) -> tuple[float, float, float]:
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return (float("nan"),) * 3
    m = float(np.mean(v)); sd = float(np.std(v))
    return m, sd, sd / math.sqrt(len(v))


def summarize(rs):
    """rs: list of (ok, n_tokens, why, (enc_ms, gen_ms, dec_ms))."""
    n = len(rs)
    p = float(np.mean([0.0 if r[0] else 1.0 for r in rs]))
    err_se = math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")
    tok, tok_std, tok_se = _mss([r[1] for r in rs])
    enc, enc_std, enc_se = _mss([r[3][0] for r in rs])
    gen, gen_std, gen_se = _mss([r[3][1] for r in rs])
    dec, dec_std, dec_se = _mss([r[3][2] for r in rs])
    forced = float(np.mean([1.0 if r[2] == "forced" else 0.0 for r in rs]))
    return {"err": p, "err_se": err_se,
            "tok": tok, "tok_std": tok_std, "tok_se": tok_se,
            "enc_ms": enc, "enc_ms_std": enc_std, "enc_ms_se": enc_se,
            "gen_ms": gen, "gen_ms_std": gen_std, "gen_ms_se": gen_se,
            "dec_ms": dec, "dec_ms_std": dec_std, "dec_ms_se": dec_se,
            "tot_ms": enc + gen + dec,
            "forced_frac": forced}


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

        for L in L_VALUES:
            ra = 1.0 - 1.0 / L
            for scheme_name, n_packets in PACKET_SCHEMES:
                k_pkt = K_BITS_TOTAL // n_packets
                name = f"BAM-{scheme_name}-L{L}"
                log("\n" + "=" * 72)
                log(f"[{model_name}] {name}  ({n_packets} packet(s) x "
                    f"{k_pkt} bits, M={2 ** k_pkt} each; g1={GAMMA} "
                    f"rACK={ra:.6f} rNACK={RHO_NACK})  — {N_TRIALS} trials")
                log("=" * 72)
                rs = []
                for i in range(N_TRIALS):
                    prompt_ids = pool[i % len(pool)]
                    # Paired randomness: identical (prompt, 16-bit message,
                    # seed) for all three schemes and every L at fixed i.
                    trial_seed = 70000 + i
                    m_true = int(np.random.RandomState(trial_seed)
                                 .randint(2 ** K_BITS_TOTAL))
                    np.random.seed(trial_seed)
                    torch.manual_seed(trial_seed)
                    t0 = time.time()
                    ok, n, why, tt = bam_packetized(
                        prompt_ids, m_true, n_packets, GAMMA, ra, RHO_NACK)
                    enc_ms, gen_ms, dec_ms = tt.per_token_ms()
                    dt = time.time() - t0
                    rs.append((ok, n, why, (enc_ms, gen_ms, dec_ms)))
                    log(f"  trial {i + 1:>4}/{N_TRIALS}: m={m_true:>5} -> "
                        f"{'OK' if ok else 'WRONG':>5} n={n:>3} ({why}) "
                        f"enc={enc_ms:.1f} gen={gen_ms:.1f} dec={dec_ms:.1f} "
                        f"ms/tok [{dt:.1f}s]")
                    if (i + 1) % GC_EVERY == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                s = summarize(rs)
                log(f"  --> err={s['err']:.4f}±{s['err_se']:.4f}(SE)  "
                    f"avg_tok={s['tok']:.2f}±{s['tok_se']:.2f}(SE)")
                log(f"      per-token: enc={s['enc_ms']:.2f}±{s['enc_ms_se']:.3f}  "
                    f"gen={s['gen_ms']:.2f}±{s['gen_ms_se']:.3f}  "
                    f"dec={s['dec_ms']:.2f}±{s['dec_ms_se']:.3f}  "
                    f"total={s['tot_ms']:.2f} ms  "
                    f"forced_frac={s['forced_frac']:.3f}")
                rows.append({
                    "model": model_name, "scheme": scheme_name,
                    "n_packets": n_packets, "k_pkt": k_pkt, "L": L, **s,
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
log(f"\n{'Model':<24} {'Scheme':<6} {'L':>6} "
    f"{'err':>8} {'err_se':>8} {'avg_tok':>8} {'tok_se':>7} "
    f"{'enc':>6} {'gen':>6} {'dec':>6} {'total':>6} {'forced':>7}")
log("-" * 110)
last_model = None
for row in all_results:
    short = row["model"].split('/')[-1]
    if short != last_model:
        if last_model is not None:
            log("-" * 110)
        last_model = short
    log(f"{short:<24} {row['scheme']:<6} {row['L']:>6} "
        f"{row['err']:>8.4f} {row['err_se']:>8.4f} "
        f"{row['tok']:>8.2f} {row['tok_se']:>7.2f} "
        f"{row['enc_ms']:>6.2f} {row['gen_ms']:>6.2f} {row['dec_ms']:>6.2f} "
        f"{row['tot_ms']:>6.2f} {row['forced_frac']:>7.3f}")

with open(OUT_CSV, "w") as f:
    f.write("model,scheme,n_packets,k_pkt,L,"
            "err_rate,err_se,avg_tokens,std_tokens,se_tokens,"
            "enc_ms,enc_ms_std,enc_ms_se,"
            "gen_ms,gen_ms_std,gen_ms_se,"
            "dec_ms,dec_ms_std,dec_ms_se,"
            "total_ms,forced_frac\n")
    for row in all_results:
        f.write(f"{row['model']},{row['scheme']},{row['n_packets']},"
                f"{row['k_pkt']},{row['L']},"
                f"{row['err']:.6f},{row['err_se']:.6f},"
                f"{row['tok']:.4f},{row['tok_std']:.4f},{row['tok_se']:.4f},"
                f"{row['enc_ms']:.4f},{row['enc_ms_std']:.4f},{row['enc_ms_se']:.4f},"
                f"{row['gen_ms']:.4f},{row['gen_ms_std']:.4f},{row['gen_ms_se']:.4f},"
                f"{row['dec_ms']:.4f},{row['dec_ms_std']:.4f},{row['dec_ms_se']:.4f},"
                f"{row['tot_ms']:.4f},{row['forced_frac']:.4f}\n")
log(f"\nWrote {OUT_CSV}")


# ── Plot: frontier (err vs avg tokens) + per-component time breakdown ───────
SCHEME_STYLE = {
    "1x24": ("#d62728", "s", "1 x 24-bit packet (M=16.8M)"),
    "2x12": ("#1f77b4", "o", "2 x 12-bit packets (M=4096)"),
    "3x8":  ("#2ca02c", "^", "3 x 8-bit packets (M=256)"),
}
COMP_COLORS = {"enc": "#8c564b", "gen": "#9467bd", "dec": "#ff7f0e"}
COMP_LABELS = {"enc": "encoding (OT, sampling rule)",
               "gen": "token generation (LLM)",
               "dec": "decoding (posterior + threshold)"}

models_in_order = [m.split('/')[-1] for m in MODEL_NAMES
                   if any(r["model"] == m for r in all_results)]
n_models = max(1, len(models_in_order))
fig, axes = plt.subplots(n_models, 2, figsize=(13.0, 5.0 * n_models),
                         squeeze=False)

for mi, short in enumerate(models_in_order):
    rows = [r for r in all_results if r["model"].split('/')[-1] == short]
    ax_f, ax_t = axes[mi][0], axes[mi][1]

    # Left: length-vs-error frontier, horizontal SE bars on length.
    for scheme, (color, marker, label) in SCHEME_STYLE.items():
        pts = sorted([(r["tok"], r["err"], r["tok_se"], r["L"])
                      for r in rows if r["scheme"] == scheme])
        if not pts:
            continue
        ax_f.errorbar([p[0] for p in pts], [p[1] for p in pts],
                      xerr=[p[2] for p in pts],
                      marker=marker, linestyle="-", color=color,
                      markersize=6.5, linewidth=1.6, capsize=2.5,
                      elinewidth=1.0, label=label)
        for tok, err, _, L in pts:
            ax_f.annotate(f"L={L}", (tok, err), textcoords="offset points",
                          xytext=(6, 4), fontsize=7, color=color)
    ax_f.set_xlabel("Average tokens", fontsize=13)
    ax_f.set_ylabel("16-bit message error rate", fontsize=13)
    ax_f.tick_params(labelsize=11)
    ax_f.set_yscale("log")
    ax_f.grid(True, which="both", alpha=0.3)
    ax_f.legend(fontsize=10)

    # Right: stacked per-token time breakdown per scheme (averaged over L;
    # component costs are L-independent, so pooling is legitimate — the SE
    # whisker on top of each stack is the SE of the TOTAL across all trials).
    schemes = [s for s in SCHEME_STYLE if any(r["scheme"] == s for r in rows)]
    x = np.arange(len(schemes))
    bottoms = np.zeros(len(schemes))
    for comp in ("gen", "enc", "dec"):
        vals = []
        for s in schemes:
            rr = [r for r in rows if r["scheme"] == s]
            vals.append(float(np.mean([r[f"{comp}_ms"] for r in rr])))
        ax_t.bar(x, vals, 0.55, bottom=bottoms,
                 color=COMP_COLORS[comp], label=COMP_LABELS[comp])
        bottoms += np.array(vals)
    tot_se = []
    for s in schemes:
        rr = [r for r in rows if r["scheme"] == s]
        ses = [math.sqrt(r["enc_ms_se"] ** 2 + r["gen_ms_se"] ** 2
                         + r["dec_ms_se"] ** 2) for r in rr]
        tot_se.append(float(np.mean(ses)))
    ax_t.errorbar(x, bottoms, yerr=tot_se, fmt="none", ecolor="black",
                  capsize=3, elinewidth=1.2)
    ax_t.set_xticks(x)
    _m_of = {nm: 2 ** (K_BITS_TOTAL // n) for nm, n in PACKET_SCHEMES}
    ax_t.set_xticklabels([f"{s}\n(M={_m_of[s]})" for s in schemes],
                         fontsize=11)
    ax_t.set_ylabel("Time per sampled token (ms)", fontsize=13)
    ax_t.tick_params(labelsize=11)
    ax_t.grid(True, axis="y", alpha=0.3)
    ax_t.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=160)
plt.savefig(OUT_PLOT.replace(".png", ".pdf"))
log(f"Wrote {OUT_PLOT} (+ .pdf)\nDone.")