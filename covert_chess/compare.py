"""
compare.py — Burnashev-ArcMark (BAM) vs Fixed-Length ArcMark vs open-source
multi-bit baselines (MPAC, BiMark, StealthInk) on C4 RealNews.
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
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from datasets import load_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARCMARK_SRC = os.environ.get(
    "ARCMARK_SRC",
    os.path.normpath(os.path.join(_THIS_DIR, "..", "arcmark_src")),
)
if ARCMARK_SRC not in sys.path:
    sys.path.insert(0, ARCMARK_SRC)
if _THIS_DIR not in sys.path:          # so ./baselines/ is importable
    sys.path.insert(0, _THIS_DIR)

from arcmark.coding import RandomLinearCode
from arcmark.config import ArcMarkConfig
from arcmark.processor import ArcMarkLogitsProcessor
from arcmark.message_decoder import decode_with_code
from arcmark.sinkhorn import extract_conditional, solve_arcmark_ot
from arcmark.side_info import SideInfoMode, compute_key_si

# ── Vendored open-source baselines (see baselines/__init__.py for provenance)
from baselines.mpac import (WatermarkLogitsProcessor as MpacLogitsProcessor,
                            WatermarkDetector as MpacDetector)
from baselines.bimark import WatermarkBimark, BimarkDetector
from baselines.stealthink import (ReweightProcessor as SIReweightProcessor,
                                  ReweightLogitsProcessor as SIReweightLogitsProcessor,
                                  DetectorProcessor as SIDetectorProcessor,
                                  generate_exact_n_tokens as si_generate_exact_n_tokens,
                                  stealthink_decode)


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAMES = [
    "unsloth/Meta-Llama-3.1-8B",
    "unsloth/Qwen3.5-9B-Base",
    "unsloth/mistral-7b-v0.3"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False
N_TRIALS   = 100 if not SMOKE_TEST else 4

FIXED_NS = [20, 30, 40, 50, 60] if not SMOKE_TEST else [32]

# ── BAM parameter tuple: BAM(gamma, rho_ACK, rho_NACK, (eps, eps_ACK), T*) ──
# Fixed across the sweep: gamma=0.5, rho_NACK=0.75, (eps, eps_ACK)=(0.4, 0.4),
# T*=MAX_TOKENS=1000. Only rho_ACK varies, tied to L via rho_ACK = 1 - 1/L,
# i.e. BAM(0.5, 1 - L^{-1}, 0.75, (0.4, 0.4), 1000) for each L below.
#L_VALUES = (
#    [2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768, 32768*4]
#    if not SMOKE_TEST else [8]
#)
L_VALUES = (
    [32768*4]
    if not SMOKE_TEST else [8]
)
GAMMA      = 0.5     # gamma  (communication-phase decision threshold g1)
RHO_NACK   = 0.75    # rho_NACK
# Each config is (g1=gamma, ra=rho_ACK=1-1/L, rn=rho_NACK, name)
BAM_CONFIGS = [
    (GAMMA, 1.0 - 1.0 / L, RHO_NACK, f"BAM-L{L}")
    for L in L_VALUES
]

EPS_NOISE  = 0.4     # eps      (communication-phase Laplace floor)
EPS_CONF = 0.4       # eps_ACK  (confirmation-phase antipodal floor)

MAX_TOKENS      = 1000
MAX_CONF_STEPS  = 120
MIN_COMM_TOKENS = 0

OUT_PLOT = "comparison_c4.png"
OUT_CSV  = "comparison_c4.csv"
OUT_PPL_PLOT = "comparison_c4_perplexity.png"

# ── Judge-text dump (for later offline LLM-as-judge, e.g. judge.py) ──────────
# We only save stego/base TEXT pairs for ONE representative BAM operating point
# (the most-reliable, highest-rho_ACK point, L=32768), capped at N_DUMP trials.
# Saving every setting would be wasteful: the text is free to capture during the
# sweep, but each saved pair is a future judge API call, so we keep it lean.
DUMP_JUDGE_TEXT = True
DUMP_SCHEME     = "BAM-L32768"            # only this scheme's pairs are saved
N_DUMP          = 100                     # cap on saved pairs (not all N_TRIALS)
OUT_JUDGE_JSONL = "judge_pairs_BAM-L32768.jsonl"

# ── Shared ArcMark core knobs ───────────────────────────────────────────────
P_FIELD            = 4
R_RESOLUTION       = 4
# 128-bit shared seed (lambda = 128). The key schedule in side_info.py now
# encodes the seed as 16 little-endian bytes, so the full 128 bits flow into
# the SHA-256 that derives (s_index, perm_seed, R_t). The previous value
# (0xA12C, 16 bits) combined with a signed-int64 packing capped the realized
# seed entropy far below the lambda = 128 claimed in the paper.
SHARED_SEED        = 0x9E3779B97F4A7C15F39CC0605CEDC834
TOP_K              = 50
SINKHORN_REG       = 0.2
SINKHORN_MAX_ITER  = 4000
SINKHORN_STOP_THR  = 1e-4
PHI                = 0.0

M_MSG  = 256
K_BITS = 8

N_PROMPTS         = 200
PROMPT_TOKEN_LEN  = 32

# ── Baseline scheme knobs ───────────────────────────────────────────────────
# Each baseline keeps the defaults of its official repo / paper unless noted.
# All embed the same K_BITS-bit payload and are swept over the SAME fixed
# lengths as Fixed-Length ArcMark (FIXED_NS), on the same prompts/seeds.
#
# MPAC (Yoo et al., NAACL 2024): green-list bias delta (distortionary),
# Position Allocation + radix-4 Colorlist, lefthash context (simple_1) — the
# paper's main configuration (gamma=0.25, r=4). delta=1.5 chosen to match the
# BiMark paper's MPAC(1.5) operating point.
MPAC_GAMMA    = 0.25
MPAC_DELTA    = 1.5
MPAC_RADIX    = 4
MPAC_SEEDING  = "simple_1"

# BiMark (Feng et al., ICML 2025): unbiased bit-flip multilayer reweighting.
# Repo generate-script defaults: prob_delta=0.2, 20 vocabulary partitions,
# 2-token context window, internal top-50. The repo draws the 20 partition
# seeds randomly at run time and shares the list with the decoder; we make
# that list deterministic from SHARED_SEED so encode/decode always agree.
BIMARK_DELTA        = 0.2
BIMARK_LAYERS       = 20
BIMARK_WINDOW       = 2
BIMARK_TOPK         = 50
BIMARK_C_KEY        = 8214793      # repo default (bit-flip key)
BIMARK_BIT_IDX_KEY  = 283519       # repo default (position key)
BIMARK_PARTITION_SEEDS = [
    int(x) for x in
    np.random.default_rng(SHARED_SEED).choice(10000, BIMARK_LAYERS,
                                              replace=False)
]

# StealthInk (Jiang et al., ICML 2025): distribution-preserving reweighting
# over randomized vocabulary permutations. Repo defaults: chunk capacity 1
# (1 bit / position => 8 positions for our 8-bit payload), 3-gram seeding
# ("simple_3"), full-vocab sampling (top_k=0 — the reweighting already zeroes
# the red mass), temperature 1.0.
STEALTHINK_CAPACITY = 1
STEALTHINK_NGRAM    = 3

BASELINE_SCHEMES = ["MPAC", "BiMark", "StealthInk"]  # order of the sweep

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


def _ppl_from_logprobs(logprob_sum: float, n: int) -> float:
    return math.exp(-logprob_sum / n) if n > 0 else float("nan")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def _mean_std_sem(vals: list[float]) -> tuple[float, float, float]:
    """Return (mean, sd, se) over NaN-filtered values; se = sd / sqrt(n)."""
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v)); sd = float(np.std(v))
    se = sd / math.sqrt(len(v)) if len(v) > 0 else float("nan")
    return m, sd, se


# ============================================================================
# Timing helpers (sec/token)
# ============================================================================
def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WallTimer:
    """CUDA-synchronized wall-clock timer for a code region."""
    def __enter__(self):
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _cuda_sync()
        self.seconds = time.perf_counter() - self.t0
        return False


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
        self.code_cache: dict[int, RandomLinearCode] = {}
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


@torch.no_grad()
def next_token_probs(input_ids: list[int]) -> torch.Tensor:
    model = CTX.model
    ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    out = model(ids, use_cache=False)
    logits = out.logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


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
# Base-model perplexity helpers (watermark OFF)
# ============================================================================
@torch.no_grad()
def baseline_logprobs(prompt_ids: list[int], n_tokens: int) -> list[float]:
    if n_tokens <= 0:
        return []
    k = ARC_CONFIG.top_k
    lm = IncrementalLM(list(prompt_ids))
    logps: list[float] = []
    try:
        for _ in range(n_tokens):
            probs = lm.probs()
            if k is not None and 0 < k < probs.numel():
                topk = torch.topk(probs, k)
                trunc = torch.zeros_like(probs)
                trunc[topk.indices] = topk.values
                trunc = trunc / trunc.sum()
                tok = int(torch.multinomial(trunc, num_samples=1).item())
            else:
                tok = int(torch.multinomial(probs, num_samples=1).item())
            logps.append(float(torch.log(probs[tok].clamp_min(1e-30)).item()))
            lm.advance(tok)
        return logps
    finally:
        lm.free()


@torch.no_grad()
def baseline_logprobs_and_tokens(prompt_ids: list[int],
                                 n_tokens: int) -> tuple[list[float], list[int], float]:
    """Like baseline_logprobs, but also returns the sampled token ids and the
    CUDA-synchronized wall-clock seconds of the generation loop.

    Used to build the length-matched no-embedding reference for perplexity and
    distinct-n, and to time un-watermarked generation (base sec/token). Same
    top-k sampling path as baseline_logprobs, so the two remain
    distribution-identical.
    """
    if n_tokens <= 0:
        return [], [], 0.0
    k = ARC_CONFIG.top_k
    with WallTimer() as tim:
        lm = IncrementalLM(list(prompt_ids))
        logps: list[float] = []
        toks: list[int] = []
        try:
            for _ in range(n_tokens):
                probs = lm.probs()
                if k is not None and 0 < k < probs.numel():
                    topk = torch.topk(probs, k)
                    trunc = torch.zeros_like(probs)
                    trunc[topk.indices] = topk.values
                    trunc = trunc / trunc.sum()
                    tok = int(torch.multinomial(trunc, num_samples=1).item())
                else:
                    tok = int(torch.multinomial(probs, num_samples=1).item())
                logps.append(float(torch.log(probs[tok].clamp_min(1e-30)).item()))
                toks.append(tok)
                lm.advance(tok)
        finally:
            lm.free()
    return logps, toks, tim.seconds


@torch.no_grad()
def score_tokens_teacher_forced(prompt_ids: list[int],
                                gen_token_ids: list[int],
                                top_k: int | None = None) -> list[float]:
    """Teacher-forced per-token log-probabilities of ``gen_token_ids``.

    When ``top_k`` is given, each position's log-probability is computed under
    the SAME top-k-truncated-and-renormalized reference distribution used by
    ArcMark/BAM and the length-matched base generation (see
    ``baseline_logprobs``), rather than the full-vocabulary softmax. This makes
    the perplexity of schemes that sample from the full distribution (MPAC,
    StealthInk, which generate with top_k=0) directly comparable to the
    top-k-referenced perplexity reported for ArcMark/BAM. A token that falls
    outside the position's top-k set is scored under the renormalized top-k
    distribution (probability ~0), i.e. it is penalized exactly as an
    out-of-support token would be for the ArcMark reference.
    """
    if len(gen_token_ids) == 0:
        return []
    model = CTX.model
    full = list(prompt_ids) + list(gen_token_ids)
    ids = torch.tensor(full, dtype=torch.long, device=model.device).unsqueeze(0)
    out = model(ids, use_cache=False)
    logits = out.logits[0].float()
    p_len = len(prompt_ids)

    if top_k is None or top_k <= 0:
        logprobs = torch.log_softmax(logits, dim=-1)
        return [float(logprobs[p_len + i - 1, tok].item())
                for i, tok in enumerate(gen_token_ids)]

    # Top-k-truncated reference: match baseline_logprobs' distribution exactly.
    logps: list[float] = []
    V = logits.shape[-1]
    k = min(top_k, V)
    for i, tok in enumerate(gen_token_ids):
        pos = p_len + i - 1
        probs = torch.softmax(logits[pos], dim=-1)
        topk = torch.topk(probs, k)
        trunc = torch.zeros_like(probs)
        trunc[topk.indices] = topk.values
        trunc = trunc / trunc.sum()
        logps.append(float(torch.log(trunc[tok].clamp_min(1e-30)).item()))
    return logps


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
    """Derive the synchronized per-token side information (s_index, perm_seed,
    R_t) for the NEXT token, from the current transcript context.

    This is the KeyGen of the paper's Eq. (key-generation): a single keyed
    SHA-256 over (secret || context) split into three disjoint blocks giving
    the channel index k_t^{(1)} (s_index), the vocabulary-permutation seed
    Lambda_t^{(2)} (perm_seed), and the posterior-matching randomness
    R_t = Rand(Lambda_t^{(3)}). Because it is derived from the shared seed and
    the shared transcript, both encoder and decoder reconstruct the identical
    R_t; there is no unsynchronized local randomness.
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
               alphabet_size: int = P_FIELD):
    """Sample a token that embeds ``symbol`` from an alphabet of size
    ``alphabet_size`` (p) via the ArcMark OT channel.

    ``alphabet_size`` is a parameter so the confirmation phase can emit through
    a genuine p=2 antipodal channel (paper: 'apply Algorithm 1 with p=2'),
    rather than reusing the p=4 communication channel.
    """
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
    base_logprob = float(torch.log(probs[token].clamp_min(1e-30)).item())
    return token, base_logprob


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
# Entropy diagnostic
# ============================================================================
def measure_entropy(prompt_ids, n_samples=5):
    ctx = list(prompt_ids); ents = []; top1s = []
    for _ in range(n_samples):
        p = next_token_probs(ctx)
        pnp = p.cpu().numpy().astype(np.float64)
        ents.append(-np.sum(pnp * np.log(pnp + 1e-20)))
        top1s.append(float(pnp.max()))
        tok = int(torch.multinomial(p, num_samples=1).item())
        ctx.append(tok)
    return float(np.mean(ents)), float(np.mean(top1s))

# ============================================================================
# BAM — communication phase likelihood (LAPLACE core, full 4-symbol)
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
    M = len(pi); q = np.empty(M)
    cdf = np.concatenate([[0.0], np.cumsum(pi)])
    for j in range(M):
        lo, hi = cdf[j], cdf[j + 1]
        width = max(hi - lo, 1e-30)
        p_u = np.zeros(P_SYM)
        for u in range(P_SYM):
            ul, uh = u / P_SYM, (u + 1) / P_SYM
            ov = max(0.0, min(hi, uh) - max(lo, ul))
            p_u[u] = ov / width
        q[j] = float((p_u * ells).sum())
    return q


# ============================================================================
# CONFIRMATION phase — Algorithm 1 (posterior matching) instantiated at p = 2
# ============================================================================
# Faithful to the paper: the confirmation phase is NOT a bespoke channel; it is
# the SAME posterior-matching machinery run with alphabet size p = 2. The two
# antipodal confirmation symbols are u_ACK = 0 (angle 0) and u_NACK = 1
# (angle pi at p = 2). The candidate-correct case transmits u_ACK, otherwise
# u_NACK; the belief rho over {u_ACK, u_NACK} is updated with the SAME
# contaminated-Laplace likelihood (eq:mixture-likelihood) evaluated at p = 2,
# with its own contamination floor eps_ACK (EPS_CONF) and its own Laplace scale
# b = pi / (p * sqrt(2)) at p = 2 (paper: b = pi / (p sqrt 2)).
P_CONF = 2                       # confirmation alphabet size (paper: p = 2)
SYM_ACK  = 0                     # u_ACK  -> angle 2*pi*0/2 = 0
SYM_NACK = 1                     # u_NACK -> angle 2*pi*1/2 = pi

# Confirmation-phase Laplace scale, at p = 2 (INDEPENDENT of the comm scale).
_B_CONF  = math.pi / (P_CONF * math.sqrt(2.0))
_Z_CONF  = 2.0 * _B_CONF * (1.0 - math.exp(-math.pi / _B_CONF))


def conf_symbol_likelihood(angle_obs: float) -> np.ndarray:
    """Contaminated-Laplace per-symbol likelihood at p = 2 over {u_ACK, u_NACK}.

    Identical functional form to per_symbol_likelihood (eq:mixture-likelihood),
    but instantiated with p = P_CONF = 2, the confirmation Laplace scale
    _B_CONF = pi/(2 sqrt 2), and the confirmation contamination floor EPS_CONF
    (= eps_ACK). Returns [ell(u_ACK), ell(u_NACK)].
    """
    signal = np.empty(P_CONF)
    for u in range(P_CONF):
        target = (2 * math.pi * u / P_CONF + PHI) % (2 * math.pi)
        d = abs(angle_obs - target) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        signal[u] = math.exp(-d / _B_CONF) / _Z_CONF
    return (1.0 - EPS_CONF) * signal + EPS_CONF / (2.0 * math.pi)


def run_comm_step(pi, m_true, emitted, lm: "IncrementalLM", base_logps: list):
    # R_t is the synchronized, transcript-derived posterior-matching
    # randomness Rand(Lambda_t^{(3)}) of the paper (Eq. codeword generation),
    # NOT an unsynchronized local draw. It is keyed by the shared seed and the
    # shared context, so the (black-box) decoder reconstructs the identical R_t
    # and therefore the identical codeword symbol u_t.
    _, _, R = side_info_for_step(emitted)
    u = posterior_match_symbol(pi, m_true, R)
    probs = lm.probs()
    x, blp = emit_token(probs, emitted, u)
    angle = read_symbol_angle(x, emitted)
    emitted.append(x)
    base_logps.append(blp)
    lm.advance(x)
    ells = per_symbol_likelihood(angle)
    q = message_likelihood(ells, pi)
    pi_new = pi * q
    s = pi_new.sum()
    pi_new = pi_new / s if s > 0 else np.ones_like(pi) / len(pi)
    return pi_new, x, emitted


def run_confirmation(true_bit, emitted, lm: "IncrementalLM", max_steps,
                     g_ack, g_nack, base_logps: list):
    """Confirmation phase = Algorithm 1 (posterior matching) run at p = 2.

    The transmitter sends the antipodal symbol chosen by true_bit
    (true_bit == 0 -> u_ACK = 0, angle 0; true_bit == 1 -> u_NACK = 1,
    angle pi at p = 2) through a genuine p = 2 ArcMark OT channel. The shared
    belief rho = [P(u_ACK), P(u_NACK)] is updated with the SAME
    contaminated-Laplace likelihood as the communication phase, evaluated at
    p = 2 with floor eps_ACK (EPS_CONF). This mirrors 'apply Algorithm 1 with
    p = 2' from the paper. Stopping: accept (u_ACK) when rho[0] >= g_ack,
    reject (u_NACK) when rho[1] >= g_nack.

    With a degenerate 2-point belief the posterior-matching inverse-CDF map
    reduces to transmitting the fixed antipodal symbol selected by true_bit, so
    the symbol is emitted directly; the mechanism is otherwise identical to
    Algorithm 1 at p = 2.
    """
    tx_symbol = SYM_ACK if true_bit == 0 else SYM_NACK
    rho = np.array([0.5, 0.5])
    used = 0
    for _ in range(max_steps):
        probs = lm.probs()
        x, blp = emit_token(probs, emitted, tx_symbol, alphabet_size=P_CONF)
        angle = read_symbol_angle(x, emitted)
        emitted.append(x)
        base_logps.append(blp)
        lm.advance(x)
        ell = conf_symbol_likelihood(angle)                # p=2 mixture-Laplace
        rho = rho * ell
        rho /= rho.sum()
        used += 1
        if rho[0] >= g_ack:  return "ACK",  used, emitted, rho
        if rho[1] >= g_nack: return "NACK", used, emitted, rho
    return ("ACK" if rho[0] >= rho[1] else "NACK"), used, emitted, rho


def burnashev_arcmark(prompt_ids, m_true, max_tokens, g1, ra, rn,
                      min_comm=MIN_COMM_TOKENS):
    pi = np.ones(M_MSG) / M_MSG
    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    base_logps: list[float] = []
    knockdown = np.ones(M_MSG)
    t = 0
    try:
        while t < max_tokens:
            pi, x, emitted = run_comm_step(pi, m_true, emitted, lm, base_logps)
            eff = pi * knockdown; eff = eff / eff.sum()
            t += 1
            if t < min_comm:
                continue
            if eff.max() >= g1:
                cand = int(eff.argmax())
                true_bit = 0 if cand == m_true else 1
                outcome, ct, emitted, _ = run_confirmation(
                    true_bit, emitted, lm, MAX_CONF_STEPS, ra, rn, base_logps)
                t += ct
                if outcome == "ACK":
                    return cand == m_true, t, "ack", base_logps, cand, emitted
                knockdown[cand] *= (1 - rn) / rn
                pi = pi * knockdown; pi /= pi.sum()
                knockdown = np.ones(M_MSG)
                if t >= max_tokens:
                    break
        eff = pi * knockdown; eff /= eff.sum()
        decoded = int(eff.argmax())
        return decoded == m_true, t, "forced", base_logps, decoded, emitted
    finally:
        lm.free()


# ============================================================================
# Fixed-Length ArcMark
# ============================================================================
def get_code(n_tokens: int) -> RandomLinearCode:
    cache = CTX.code_cache
    code = cache.get(n_tokens)
    if code is None:
        # RandomLinearCode.build seeds a torch.Generator, which requires a
        # value fitting in int64. The linear-code seed only needs to be
        # deterministic and shared (it is not the cryptographic seed), so we
        # fold the 128-bit SHARED_SEED down to 63 bits.
        code_seed = (SHARED_SEED + 42) & ((1 << 63) - 1)
        code = RandomLinearCode.build(
            num_messages=M_MSG,
            codeword_length=n_tokens,
            alphabet_size=P_FIELD,
            seed=code_seed,
        )
        cache[n_tokens] = code
    return code


@torch.no_grad()
def fixed_length_trial(prompt_ids, payload_int, n_tokens):
    """Returns (ok, n_used, ber, decoded, gen_ids, base_logps, gen_sec, dec_sec)."""
    model = CTX.model
    tokenizer = CTX.tokenizer
    vocab_size = CTX.vocab_size
    code = get_code(n_tokens)
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
    with WallTimer() as tim_gen:
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[proc],
            **gen_kwargs,
        )
    gen_tokens = output[0, prompt_len:]

    gen_ids_list = gen_tokens.detach().cpu().tolist()
    base_logps = score_tokens_teacher_forced(prompt_ids, gen_ids_list)

    n_got = gen_tokens.shape[0]
    if n_got < n_tokens:
        log(f"    [warn] generated {n_got} tokens, expected {n_tokens}; "
            f"decoding against first {n_got} codeword columns")
        from arcmark.message_decoder import decode_message
        with WallTimer() as tim_dec:
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
        decoded = int(result.message_idx)
        return (decoded == payload_int, n_got,
                bit_error_rate(decoded, payload_int), decoded, gen_ids_list,
                base_logps, tim_gen.seconds, tim_dec.seconds)

    with WallTimer() as tim_dec:
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
    decoded = int(result.message_idx)
    return (decoded == payload_int, n_tokens,
            bit_error_rate(decoded, payload_int), decoded, gen_ids_list,
            base_logps, tim_gen.seconds, tim_dec.seconds)


# ============================================================================
# Baseline trials: MPAC, BiMark, StealthInk
# ============================================================================
# Shared trial contract (same as fixed_length_trial):
#   (prompt_ids, payload_int, n_tokens) ->
#   (ok, n_used, ber, decoded, gen_ids, base_logps, gen_sec, dec_sec)
# Each scheme embeds the K_BITS-bit payload with its OWN machinery and decodes
# with its OWN detector; base_logps is the same teacher-forced scoring under
# the unmodified base model as used for ArcMark (for perplexity).

@torch.no_grad()
def mpac_trial(prompt_ids, payload_int, n_tokens):
    model = CTX.model
    tokenizer = CTX.tokenizer
    binary_msg = format(payload_int, f"0{K_BITS}b")

    proc = MpacLogitsProcessor(
        vocab=list(range(CTX.vocab_size)),
        gamma=MPAC_GAMMA,
        delta=MPAC_DELTA,
        seeding_scheme=MPAC_SEEDING,
        base=MPAC_RADIX,
        message_length=K_BITS,
        code_length=K_BITS,      # == message_length: plain payload, no ECC
        device=model.device,
    )
    proc.set_message(binary_msg)

    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    with WallTimer() as tim_gen:
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=True,
            temperature=1.0,
            top_k=0,             # MPAC pipeline style: pure sampling of biased dist
            min_new_tokens=n_tokens,
            max_new_tokens=n_tokens,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[proc],
        )
    gen_ids = output[0, len(prompt_ids):].detach().cpu().tolist()
    positions = proc.flush_position()[0]   # gold bit positions (metric only)

    base_logps = score_tokens_teacher_forced(prompt_ids, gen_ids, top_k=TOP_K)

    detector = MpacDetector(
        vocab=list(range(CTX.vocab_size)),
        gamma=MPAC_GAMMA,
        delta=MPAC_DELTA,
        seeding_scheme=MPAC_SEEDING,
        base=MPAC_RADIX,
        message_length=K_BITS,
        code_length=K_BITS,
        device=model.device,
        tokenizer=tokenizer,
        normalizers=[],
        ignore_repeated_ngrams=False,
    )
    with WallTimer() as tim_dec:
        res = detector.detect(
            tokenized_text=torch.tensor(gen_ids, dtype=torch.long,
                                        device=model.device),
            message=binary_msg,
            position=positions,
            return_prediction=False,
        )
    ok = bool(res["bit_match"])            # MPAC's own exact-match accounting
    ber = 1.0 - float(res["bit_acc"])      # MPAC's own bit accuracy
    pred_digits = res["pred_message"]      # radix-MPAC_RADIX digit string
    decoded = min(int(pred_digits, MPAC_RADIX), M_MSG - 1)
    return (ok, len(gen_ids), ber, decoded, gen_ids, base_logps,
            tim_gen.seconds, tim_dec.seconds)


@torch.no_grad()
def bimark_trial(prompt_ids, payload_int, n_tokens):
    model = CTX.model
    tokenizer = CTX.tokenizer
    bits = format(payload_int, f"0{K_BITS}b")

    # Fresh processor per trial (it keeps per-generation state: cnt, hist).
    proc = WatermarkBimark(
        tokenizer=tokenizer,
        vocab_size=CTX.vocab_size,
        device=model.device,
        top_k=BIMARK_TOPK,
        partition_seeds=list(BIMARK_PARTITION_SEEDS),
        c_key=BIMARK_C_KEY,
        bit_idx_key=BIMARK_BIT_IDX_KEY,
        delta=BIMARK_DELTA,
        window_size=BIMARK_WINDOW,
        bits=bits,
    )
    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    with WallTimer() as tim_gen:
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=True,
            temperature=1.0,
            top_k=0,             # processor already restricts to its top-50
            min_new_tokens=n_tokens,
            max_new_tokens=n_tokens,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[proc],
        )
    gen_ids = output[0, len(prompt_ids):].detach().cpu().tolist()

    base_logps = score_tokens_teacher_forced(prompt_ids, gen_ids)

    det = BimarkDetector(tokenizer=tokenizer, vocab_size=CTX.vocab_size,
                         window_size=BIMARK_WINDOW, gamma=0.5)
    with WallTimer() as tim_dec:
        (_, _, _, _, _, _, decode_bits, hit, hit_rate) = \
            det.decode_bimark_multibit_watermark(
                torch.tensor(gen_ids, dtype=torch.long),
                list(BIMARK_PARTITION_SEEDS),
                BIMARK_C_KEY, BIMARK_BIT_IDX_KEY,
                bits, stride=max(1, len(gen_ids)),
            )
    # The last stride bucket accumulates the full sequence; ties decode to 'x'
    # and count as wrong under BiMark's own `hit` accounting.
    ok = (hit[-1] == K_BITS)
    ber = 1.0 - float(hit_rate[-1])
    decoded_bits = decode_bits[-1].replace("x", "0")   # for logging only
    decoded = int(decoded_bits, 2) if decoded_bits else 0
    return (ok, len(gen_ids), ber, decoded, gen_ids, base_logps,
            tim_gen.seconds, tim_dec.seconds)


@torch.no_grad()
def stealthink_trial(prompt_ids, payload_int, n_tokens, trial_seed):
    model = CTX.model
    tokenizer = CTX.tokenizer
    num_value = 2 ** STEALTHINK_CAPACITY
    R = 1.0 / num_value
    converted_msg_length = K_BITS // STEALTHINK_CAPACITY
    max_len_bits = len(bin(num_value - 1)[2:])
    binary_mapping = {g: bin(g)[2:].zfill(max_len_bits) for g in range(num_value)}

    bits = format(payload_int, f"0{K_BITS}b")
    # per-position message values (capacity bits per position, MSB first)
    embedded_message = [
        int(bits[p * STEALTHINK_CAPACITY:(p + 1) * STEALTHINK_CAPACITY], 2)
        for p in range(converted_msg_length)
    ]

    # NOTE: vocab must span the model's LOGITS dimension (CTX.vocab_size), not
    # the tokenizer vocab, because the generation-side permutation is drawn
    # over probs.shape[-1]; the detector permutation must match it exactly.
    vocab = list(range(CTX.vocab_size))
    rp = SIReweightProcessor(vocab=vocab)
    dp = SIDetectorProcessor(vocab=vocab)
    lp = SIReweightLogitsProcessor(
        rp, embedded_message=embedded_message,
        n_gram_len=STEALTHINK_NGRAM, R=R,
        converted_msg_length=converted_msg_length,
        seen_seeds=set(),
    )
    inputs = torch.tensor(prompt_ids, dtype=torch.long,
                          device=model.device).unsqueeze(0)
    with WallTimer() as tim_gen:
        seq = si_generate_exact_n_tokens(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            logits_processor=LogitsProcessorList([lp]),
            n_new_tokens=n_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,             # reweighting already zeroes the red mass
            eos_id=tokenizer.eos_token_id,
            soft_eos_penalty=0.0,
        )
    gen_ids = seq[0, len(prompt_ids):].detach().cpu().tolist()

    base_logps = score_tokens_teacher_forced(prompt_ids, gen_ids, top_k=TOP_K)

    with WallTimer() as tim_dec:
        _, _, msg = stealthink_decode(
            seq, len(prompt_ids), n_tokens, STEALTHINK_NGRAM,
            rp, dp, converted_msg_length, num_value, R,
        )
    # BER via StealthInk's own tie-averaged correct-bit accounting.
    total_correct_bits = 0.0
    for pos in range(converted_msg_length):
        correct_bits = 0
        for cand in msg[pos]:
            ham = sum(c1 != c2 for c1, c2 in
                      zip(binary_mapping[cand],
                          binary_mapping[embedded_message[pos]]))
            correct_bits += STEALTHINK_CAPACITY - ham
        total_correct_bits += correct_bits / max(1, len(msg[pos]))
    ber = 1.0 - total_correct_bits / float(K_BITS)
    # Message error via a single tie-broken decode (uniform among each
    # position's argmin set, seeded per trial for reproducibility).
    rng = np.random.RandomState(trial_seed)
    decoded_positions = [
        cands[0] if len(cands) == 1 else int(cands[int(rng.randint(len(cands)))])
        for cands in msg
    ]
    decoded_bits = "".join(binary_mapping[v] for v in decoded_positions)
    decoded = int(decoded_bits, 2)
    ok = (decoded == payload_int)
    return (ok, len(gen_ids), ber, decoded, gen_ids, base_logps,
            tim_gen.seconds, tim_dec.seconds)


# ============================================================================
# Metrics: bit-error rate, distinct-n  (steganalysis F1 removed)
# ============================================================================
def bit_error_rate(decoded_idx: int, true_idx: int, k_bits: int = K_BITS) -> float:
    """Fraction of the k_bits payload bits that are wrong (Hamming / k_bits).

    The message index in [0, 2^k_bits) is the k_bits-bit payload; BER is the
    per-bit error, reported alongside (not instead of) full-message error.
    """
    diff = int(decoded_idx) ^ int(true_idx)
    hamming = bin(diff & ((1 << k_bits) - 1)).count("1")
    return hamming / float(k_bits)


def distinct_n(token_ids: list[int], n: int) -> float:
    """distinct-n = |unique n-grams| / |n-grams| over a token sequence.

    Returns NaN when the sequence is too short to contain an n-gram.
    """
    if len(token_ids) < n:
        return float("nan")
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return float("nan")
    return len(set(grams)) / float(len(grams))


def distinct_234(token_ids: list[int]) -> tuple[float, float, float]:
    return (distinct_n(token_ids, 2),
            distinct_n(token_ids, 3),
            distinct_n(token_ids, 4))


def dump_judge_pairs(path: str,
                     records: list[dict]) -> None:
    """Append stego/base TEXT pairs to a JSONL for later offline LLM judging.

    Each record: {idx, prompt, stego_text, base_text, n_tokens, message,
    decoded, correct}. Text is detokenized here so the judge script needs no
    tokenizer. Called for ONE BAM operating point only (see DUMP_SCHEME).
    """
    import json
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rs):
    errs = float(np.mean([not r[0] for r in rs]))
    tok_vals = [r[1] for r in rs]
    toks = float(np.mean(tok_vals))
    std  = float(np.std(tok_vals))
    sem  = std / math.sqrt(len(tok_vals)) if tok_vals else float("nan")
    return errs, toks, std, sem


# ============================================================================
# Shared per-trial metric bookkeeping
# ============================================================================
class TrialMetrics:
    """Accumulates the per-trial metrics shared by every scheme sweep."""

    def __init__(self):
        self.rs = []                 # (ok, n_tokens[, ...])
        self.ppl_wm = []
        self.ppl_base = []
        self.ber = []
        self.d_wm = {2: [], 3: [], 4: []}
        self.d_bs = {2: [], 3: [], 4: []}
        self.sec_tok_gen = []        # watermarked generation sec/token
        self.dec_sec = []            # decode wall-clock seconds
        self.sec_tok_base = []       # unwatermarked generation sec/token

    def add(self, ok, n_used, ber, base_logps, stego_ids, base_ids,
            ppl_base, gen_sec, dec_sec, base_sec, extra=None):
        self.rs.append((ok, n_used) + (tuple(extra) if extra else ()))
        self.ppl_wm.append(_ppl_from_logprobs(sum(base_logps), len(base_logps)))
        self.ppl_base.append(ppl_base)
        self.ber.append(ber)
        for nn, val in zip((2, 3, 4), distinct_234(stego_ids)):
            if not math.isnan(val): self.d_wm[nn].append(val)
        for nn, val in zip((2, 3, 4), distinct_234(base_ids)):
            if not math.isnan(val): self.d_bs[nn].append(val)
        if n_used > 0:
            self.sec_tok_gen.append(gen_sec / n_used)
            self.sec_tok_base.append(base_sec / n_used if base_sec > 0
                                     else float("nan"))
        self.dec_sec.append(dec_sec)

    def row(self, model_name, scheme, kind):
        e, t, s, t_se = summarize(self.rs)
        ber_mean, _ = _mean_std(self.ber)
        ppl_wm_mean, ppl_wm_std, ppl_wm_se = _mean_std_sem(self.ppl_wm)
        ppl_base_mean, ppl_base_std, ppl_base_se = _mean_std_sem(self.ppl_base)
        dwm = {nn: _mean_std(self.d_wm[nn])[0] for nn in (2, 3, 4)}
        dbs = {nn: _mean_std(self.d_bs[nn])[0] for nn in (2, 3, 4)}
        st_mean, _, st_se = _mean_std_sem(self.sec_tok_gen)
        sb_mean, _, sb_se = _mean_std_sem(self.sec_tok_base)
        dec_mean, _ = _mean_std(self.dec_sec)
        return {
            "model": model_name, "scheme": scheme, "kind": kind,
            "err": e, "ber": ber_mean, "tok": t, "std": s, "tok_se": t_se,
            "ppl_wm_mean": ppl_wm_mean, "ppl_wm_std": ppl_wm_std,
            "ppl_wm_se": ppl_wm_se,
            "ppl_base_mean": ppl_base_mean, "ppl_base_std": ppl_base_std,
            "ppl_base_se": ppl_base_se,
            "dist2_wm": dwm[2], "dist3_wm": dwm[3], "dist4_wm": dwm[4],
            "dist2_base": dbs[2], "dist3_base": dbs[3], "dist4_base": dbs[4],
            "sec_tok": st_mean, "sec_tok_se": st_se,
            "dec_sec": dec_mean,
            "base_sec_tok": sb_mean, "base_sec_tok_se": sb_se,
        }

    def log_summary(self):
        e, t, s, t_se = summarize(self.rs)
        ber_mean, _ = _mean_std(self.ber)
        ppl_wm_mean, _, ppl_wm_se = _mean_std_sem(self.ppl_wm)
        ppl_base_mean, _, ppl_base_se = _mean_std_sem(self.ppl_base)
        dwm = {nn: _mean_std(self.d_wm[nn])[0] for nn in (2, 3, 4)}
        dbs = {nn: _mean_std(self.d_bs[nn])[0] for nn in (2, 3, 4)}
        st_mean, _, st_se = _mean_std_sem(self.sec_tok_gen)
        sb_mean, _, _ = _mean_std_sem(self.sec_tok_base)
        dec_mean, _ = _mean_std(self.dec_sec)
        log(f"  --> err_rate={e:.4f}  ber={ber_mean:.4f}  "
            f"avg_tok={t:.2f}±{t_se:.2f}(SE)  std={s:.2f}  "
            f"ppl_wm={ppl_wm_mean:.2f}±{ppl_wm_se:.2f}(SE)  "
            f"ppl_base={ppl_base_mean:.2f}±{ppl_base_se:.2f}(SE)")
        log(f"      distinct-2/3/4 wm={dwm[2]:.3f}/{dwm[3]:.3f}/{dwm[4]:.3f}  "
            f"base={dbs[2]:.3f}/{dbs[3]:.3f}/{dbs[4]:.3f}  "
            f"sec/tok gen={st_mean:.4f}±{st_se:.4f}(SE) base={sb_mean:.4f}  "
            f"decode={dec_mean:.4f}s")


# ============================================================================
# Fixed-length sweep runner (shared by FL ArcMark and the three baselines)
# ============================================================================
def run_fixed_scheme(model_name: str, scheme_label: str, kind: str,
                     n_tok: int, seed_base: int, trial_fn) -> dict:
    """trial_fn(prompt_ids, payload, n_tok, trial_seed) ->
    (ok, n_used, ber, decoded, gen_ids, base_logps, gen_sec, dec_sec)."""
    pool = CTX.prompt_pool
    log("\n" + "=" * 72)
    log(f"[{model_name}] {scheme_label} n={n_tok}  — {N_TRIALS} trials")
    log("=" * 72)
    tm = TrialMetrics()
    for i in range(N_TRIALS):
        prompt_ids = pool[i % len(pool)]
        payload = i % (2 ** K_BITS)
        trial_seed = seed_base + i * 10 + n_tok
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        t0 = time.time()
        (ok, n_used, ber, decoded, stego_ids, base_logps,
         gen_sec, dec_sec) = trial_fn(prompt_ids, payload, n_tok, trial_seed)
        ppl_wm = _ppl_from_logprobs(sum(base_logps), len(base_logps))
        # One length-matched base generation reused for ppl, dist-n, base timing.
        bl, base_ids, base_sec = baseline_logprobs_and_tokens(
            prompt_ids, len(base_logps))
        ppl_base = _ppl_from_logprobs(sum(bl), len(bl))
        dt = time.time() - t0
        tm.add(ok, n_used, ber, base_logps, stego_ids, base_ids,
               ppl_base, gen_sec, dec_sec, base_sec)
        log(f"  trial {i + 1:>3}/{N_TRIALS}: -> "
            f"{'OK' if ok else 'WRONG':>5} ber={ber:.3f} "
            f"ppl_wm={ppl_wm:.2f} ppl_base={ppl_base:.2f} "
            f"gen={gen_sec:.2f}s dec={dec_sec:.3f}s [{dt:.1f}s]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tm.log_summary()
    return tm.row(model_name, f"{scheme_label} n={n_tok}", kind)


# trial_fn adapters unifying the (prompt, payload, n, seed) signature
def _fl_adapter(prompt_ids, payload, n_tok, trial_seed):
    return fixed_length_trial(prompt_ids, payload, n_tok)


def _mpac_adapter(prompt_ids, payload, n_tok, trial_seed):
    return mpac_trial(prompt_ids, payload, n_tok)


def _bimark_adapter(prompt_ids, payload, n_tok, trial_seed):
    return bimark_trial(prompt_ids, payload, n_tok)


def _stealthink_adapter(prompt_ids, payload, n_tok, trial_seed):
    return stealthink_trial(prompt_ids, payload, n_tok, trial_seed)


# (scheme label, row kind, per-scheme seed base, adapter)
FIXED_SCHEME_TABLE = [
    ("FL",         "FL",         60000, _fl_adapter),
    ("MPAC",       "MPAC",       70000, _mpac_adapter),
    ("BiMark",     "BIMARK",     80000, _bimark_adapter),
    ("StealthInk", "STEALTHINK", 90000, _stealthink_adapter),
]


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

        log("\n" + "=" * 72)
        log(f"[{model_name}] Entropy diagnostic (first 3 prompts):")
        log("=" * 72)
        for i in range(min(3, len(pool))):
            H, top1 = measure_entropy(pool[i])
            log(f"  prompt {i}: H={H:.2f} nats  top-1={top1:.3f}")

        # Warm-up: untimed short generations so the first timed trial does not
        # absorb CUDA context/kernel-compilation overhead.
        log(f"[{model_name}] warm-up generations (untimed) ...")
        for _ in range(2):
            baseline_logprobs(pool[0], 16)
        _cuda_sync()

        # ── BAM (variable-length) sweep ────────────────────────────────────
        for (g1, ra, rn, name) in BAM_CONFIGS:
            log("\n" + "=" * 72)
            log(f"[{model_name}] {name}  (g1={g1} rACK={ra} rNACK={rn})"
                f"  — {N_TRIALS} trials")
            log(f"  confirmation: posterior matching at p={P_CONF} "
                f"(u_ACK=sym{SYM_ACK} angle0, u_NACK=sym{SYM_NACK} anglePi), "
                f"eps_ACK={EPS_CONF}, b_conf={_B_CONF:.4f}")
            log("=" * 72)
            tm = TrialMetrics()
            judge_records: list[dict] = []   # only filled if name == DUMP_SCHEME
            for i in range(N_TRIALS):
                prompt_ids = pool[i % len(pool)]
                m_true = int(np.random.RandomState(50000 + i).randint(M_MSG))
                np.random.seed(50000 + i)
                torch.manual_seed(50000 + i)
                t0 = time.time()
                # BAM generation is the whole interactive loop (decoding is
                # integrated: the belief update IS the decoder) -> dec_sec=0.
                with WallTimer() as tim_gen:
                    ok, n, why, base_logps, decoded, stego_ids = \
                        burnashev_arcmark(prompt_ids, m_true, MAX_TOKENS,
                                          g1, ra, rn)
                ppl_wm = _ppl_from_logprobs(sum(base_logps), len(base_logps))
                bl, base_ids, base_sec = baseline_logprobs_and_tokens(
                    prompt_ids, len(base_logps))
                ppl_base = _ppl_from_logprobs(sum(bl), len(bl))
                ber = bit_error_rate(decoded, m_true)
                dt = time.time() - t0
                tm.add(ok, n, ber, base_logps, stego_ids, base_ids,
                       ppl_base, tim_gen.seconds, 0.0, base_sec, extra=(why, dt))
                if (DUMP_JUDGE_TEXT and name == DUMP_SCHEME
                        and len(judge_records) < N_DUMP
                        and len(stego_ids) >= 2 and len(base_ids) >= 2):
                    judge_records.append({
                        "idx": i,
                        "model": model_name,
                        "prompt": CTX.tokenizer.decode(prompt_ids),
                        "stego_text": CTX.tokenizer.decode(stego_ids),
                        "base_text": CTX.tokenizer.decode(base_ids),
                        "n_tokens": len(stego_ids),
                        "message": int(m_true),
                        "decoded": int(decoded),
                        "correct": bool(ok),
                    })
                log(f"  trial {i + 1:>3}/{N_TRIALS}: m={m_true:>3} -> "
                    f"{'OK' if ok else 'WRONG':>5} n={n:>3} ({why}) "
                    f"ber={ber:.3f} ppl_wm={ppl_wm:.2f} ppl_base={ppl_base:.2f} "
                    f"gen={tim_gen.seconds:.1f}s [{dt:.1f}s]")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            tm.log_summary()
            rows.append(tm.row(model_name, name, "BAM"))
            if DUMP_JUDGE_TEXT and name == DUMP_SCHEME and judge_records:
                dump_judge_pairs(OUT_JUDGE_JSONL, judge_records)
                log(f"  [judge] wrote {len(judge_records)} stego/base text pairs "
                    f"-> {OUT_JUDGE_JSONL}")

        # ── Fixed-length sweeps: FL ArcMark + MPAC + BiMark + StealthInk ──
        for (label, kind, seed_base, adapter) in FIXED_SCHEME_TABLE:
            for n_tok in FIXED_NS:
                rows.append(run_fixed_scheme(model_name, label, kind,
                                             n_tok, seed_base, adapter))
    finally:
        CTX.teardown()
        CTX = None
    return rows


# ============================================================================
# Run all models
# ============================================================================
all_results: list[dict] = []

# Fresh judge-pair file each run (dump_judge_pairs appends per matching scheme).
if DUMP_JUDGE_TEXT:
    open(OUT_JUDGE_JSONL, "w").close()

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
# Combined report
# ============================================================================
log("\n" + "=" * 72); log("COMBINED SUMMARY (all models)"); log("=" * 72)
log(f"\n{'Model':<24} {'Scheme':<16} {'kind':<10} "
    f"{'msg_err':>8} {'ber':>7} {'avg_tok':>8} "
    f"{'ppl_wm':>8} {'ppl_base':>8} {'dppl':>6} "
    f"{'d2_wm':>6} {'d2_bse':>7} {'s/tok':>8} {'base_s/tok':>10} {'dec_s':>7}")
log("-" * 148)
last_model = None
for row in all_results:
    short = row["model"].split('/')[-1]
    if short != last_model:
        if last_model is not None:
            log("-" * 148)
        last_model = short
    dppl = row["ppl_wm_mean"] - row["ppl_base_mean"]
    log(f"{short:<24} {row['scheme']:<16} {row['kind']:<10} "
        f"{row['err']:>8.4f} {row.get('ber', float('nan')):>7.4f} "
        f"{row['tok']:>8.2f} "
        f"{row['ppl_wm_mean']:>8.2f} {row['ppl_base_mean']:>8.2f} {dppl:>+6.2f} "
        f"{row.get('dist2_wm', float('nan')):>6.3f} "
        f"{row.get('dist2_base', float('nan')):>7.3f} "
        f"{row.get('sec_tok', float('nan')):>8.4f} "
        f"{row.get('base_sec_tok', float('nan')):>10.4f} "
        f"{row.get('dec_sec', float('nan')):>7.4f}")

with open(OUT_CSV, "w") as f:
    f.write("model,scheme,kind,msg_err_rate,bit_err_rate,"
            "avg_tokens,std_tokens,se_tokens,"
            "ppl_wm_mean,ppl_wm_std,ppl_wm_se,"
            "ppl_base_mean,ppl_base_std,ppl_base_se,ppl_delta,"
            "dist2_wm,dist3_wm,dist4_wm,dist2_base,dist3_base,dist4_base,"
            "sec_per_token,sec_per_token_se,decode_sec,"
            "base_sec_per_token,base_sec_per_token_se\n")
    for row in all_results:
        dppl = row["ppl_wm_mean"] - row["ppl_base_mean"]
        g = lambda key: row.get(key, float("nan"))
        f.write(f"{row['model']},{row['scheme']},{row['kind']},"
                f"{row['err']:.6f},{g('ber'):.6f},"
                f"{row['tok']:.4f},{row['std']:.4f},{g('tok_se'):.4f},"
                f"{row['ppl_wm_mean']:.4f},{row['ppl_wm_std']:.4f},{g('ppl_wm_se'):.4f},"
                f"{row['ppl_base_mean']:.4f},{row['ppl_base_std']:.4f},"
                f"{g('ppl_base_se'):.4f},{dppl:.4f},"
                f"{g('dist2_wm'):.4f},{g('dist3_wm'):.4f},{g('dist4_wm'):.4f},"
                f"{g('dist2_base'):.4f},{g('dist3_base'):.4f},{g('dist4_base'):.4f},"
                f"{g('sec_tok'):.6f},{g('sec_tok_se'):.6f},{g('dec_sec'):.6f},"
                f"{g('base_sec_tok'):.6f},{g('base_sec_tok_se'):.6f}\n")
log(f"\nWrote {OUT_CSV}")
log("Note: BAM decode_sec is 0 by construction (belief update is the decoder,"
    " integrated in the timed generation loop).")


# ── Plot 1: error rate vs avg tokens ────────────────────────────────────────
models_in_order = [m.split('/')[-1] for m in MODEL_NAMES
                   if any(r["model"] == m for r in all_results)]
n_models = max(1, len(models_in_order))
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
axes = axes[0]

SERIES_STYLE = {
    "BAM":        ("s-", "red",       "BAM"),
    "FL":         ("o-", "blue",      "Fixed-Length"),
    "MPAC":       ("^-", "green",     "MPAC"),
    "BIMARK":     ("v-", "purple",    "BiMark"),
    "STEALTHINK": ("D-", "darkorange", "StealthInk"),
}

for ax, short in zip(axes, models_in_order):
    rows = [r for r in all_results if r["model"].split('/')[-1] == short]
    for kind, (fmt, color, label) in SERIES_STYLE.items():
        pts = [(r["tok"], r["err"], r["scheme"]) for r in rows
               if r["kind"] == kind]
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt,
                color=color, markersize=8, label=label)
        if kind == "BAM":
            for tt, ee, nm in pts:
                ax.annotate(nm, (tt, ee), textcoords="offset points",
                            xytext=(7, 5), fontsize=8)
    ax.set_xlabel("Average tokens"); ax.set_ylabel("Error rate")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(short)

fig.suptitle(f"BAM vs fixed-length schemes on C4 RealNews (N={N_TRIALS})")
plt.tight_layout(); plt.savefig(OUT_PLOT, dpi=140)
log(f"Wrote {OUT_PLOT}")


# ── Plot 2: perplexity ──────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, n_models, figsize=(9 * n_models, 5), squeeze=False)
axes2 = axes2[0]

for ax, short in zip(axes2, models_in_order):
    rows = [r for r in all_results if r["model"].split('/')[-1] == short]
    labels   = [r["scheme"] for r in rows]
    wm_mean  = [r["ppl_wm_mean"] for r in rows]
    wm_std   = [r["ppl_wm_std"] for r in rows]
    bse_mean = [r["ppl_base_mean"] for r in rows]
    bse_std  = [r["ppl_base_std"] for r in rows]
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, wm_mean,  w, yerr=wm_std,  capsize=3,
           color="seagreen", label="watermarked")
    ax.bar(x + w / 2, bse_mean, w, yerr=bse_std, capsize=3,
           color="darkgray", label="base (length-matched)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right",
                                         fontsize=7)
    ax.set_ylabel("Perplexity (base model)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    ax.set_title(short)

fig2.suptitle(f"Perplexity: watermarked vs length-matched baseline (N={N_TRIALS})")
plt.tight_layout(); plt.savefig(OUT_PPL_PLOT, dpi=140)
log(f"Wrote {OUT_PPL_PLOT}\nDone.")