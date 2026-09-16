"""
compare_bam.py — Burnashev-ArcMark (BAM) protocol error performance on C4
RealNews.

Reduced from compare.py: keeps ONLY the BAM variable-length sweep and its
error metrics (message error rate, bit error rate, average tokens, ACK vs
forced stopping). Removed: Fixed-Length ArcMark, MPAC / BiMark / StealthInk
baselines, perplexity, distinct-n, timing, and the judge-text dump.

Confirmation phase (unchanged): the 1-bit ACK/NACK confirmation is a genuine
2-symbol ANTIPODAL channel with its own clean 2-hypothesis likelihood and its
own noise floor EPS_CONF.
"""

from __future__ import annotations

import math
import os
import sys
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
    #"unsloth/mistral-7b-v0.3"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False
N_TRIALS   = 100 if not SMOKE_TEST else 4

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

OUT_PLOT = "bam_c4.png"
OUT_CSV  = "bam_c4.csv"

# ── Shared ArcMark core knobs ───────────────────────────────────────────────
P_FIELD            = 4
R_RESOLUTION       = 4
# 128-bit shared seed (lambda = 128). The key schedule in side_info.py
# encodes the seed as 16 little-endian bytes, so the full 128 bits flow into
# the SHA-256 that derives (s_index, perm_seed, R_t).
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


def _mean_std(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


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
               alphabet_size: int = P_FIELD) -> int:
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
    return int(torch.multinomial(cond, num_samples=1).item())


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


def run_comm_step(pi, m_true, emitted, lm: "IncrementalLM"):
    # R_t is the synchronized, transcript-derived posterior-matching
    # randomness Rand(Lambda_t^{(3)}) of the paper (Eq. codeword generation),
    # NOT an unsynchronized local draw. It is keyed by the shared seed and the
    # shared context, so the (black-box) decoder reconstructs the identical R_t
    # and therefore the identical codeword symbol u_t.
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
    return pi_new, x, emitted


def run_confirmation(true_bit, emitted, lm: "IncrementalLM", max_steps,
                     g_ack, g_nack):
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
        x = emit_token(probs, emitted, tx_symbol, alphabet_size=P_CONF)
        angle = read_symbol_angle(x, emitted)
        emitted.append(x)
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
    """Returns (ok, n_tokens, why, decoded, emitted)."""
    pi = np.ones(M_MSG) / M_MSG
    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    knockdown = np.ones(M_MSG)
    t = 0
    try:
        while t < max_tokens:
            pi, x, emitted = run_comm_step(pi, m_true, emitted, lm)
            eff = pi * knockdown; eff = eff / eff.sum()
            t += 1
            if t < min_comm:
                continue
            if eff.max() >= g1:
                cand = int(eff.argmax())
                true_bit = 0 if cand == m_true else 1
                outcome, ct, emitted, _ = run_confirmation(
                    true_bit, emitted, lm, MAX_CONF_STEPS, ra, rn)
                t += ct
                if outcome == "ACK":
                    return cand == m_true, t, "ack", cand, emitted
                knockdown[cand] *= (1 - rn) / rn
                pi = pi * knockdown; pi /= pi.sum()
                knockdown = np.ones(M_MSG)
                if t >= max_tokens:
                    break
        eff = pi * knockdown; eff /= eff.sum()
        decoded = int(eff.argmax())
        return decoded == m_true, t, "forced", decoded, emitted
    finally:
        lm.free()


# ============================================================================
# Metrics
# ============================================================================
def bit_error_rate(decoded_idx: int, true_idx: int, k_bits: int = K_BITS) -> float:
    """Fraction of the k_bits payload bits that are wrong (Hamming / k_bits).

    The message index in [0, 2^k_bits) is the k_bits-bit payload; BER is the
    per-bit error, reported alongside (not instead of) full-message error.
    """
    diff = int(decoded_idx) ^ int(true_idx)
    hamming = bin(diff & ((1 << k_bits) - 1)).count("1")
    return hamming / float(k_bits)


class TrialMetrics:
    """Accumulates the per-trial BAM error metrics."""

    def __init__(self):
        self.ok: list[bool] = []
        self.tok: list[int] = []
        self.ber: list[float] = []
        self.why: list[str] = []

    def add(self, ok, n_used, ber, why):
        self.ok.append(bool(ok))
        self.tok.append(int(n_used))
        self.ber.append(float(ber))
        self.why.append(why)

    def row(self, model_name, scheme):
        n = len(self.ok)
        err = float(np.mean([not o for o in self.ok])) if n else float("nan")
        tok_mean, tok_std = _mean_std([float(t) for t in self.tok])
        tok_se = tok_std / math.sqrt(n) if n else float("nan")
        ber_mean, _ = _mean_std(self.ber)
        forced = float(np.mean([w == "forced" for w in self.why])) if n else float("nan")
        return {
            "model": model_name, "scheme": scheme,
            "err": err, "ber": ber_mean,
            "tok": tok_mean, "std": tok_std, "tok_se": tok_se,
            "forced_frac": forced,
        }

    def log_summary(self, model_name, scheme):
        r = self.row(model_name, scheme)
        log(f"  --> err_rate={r['err']:.4f}  ber={r['ber']:.4f}  "
            f"avg_tok={r['tok']:.2f}±{r['tok_se']:.2f}(SE)  std={r['std']:.2f}  "
            f"forced_frac={r['forced_frac']:.3f}")


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

        for (g1, ra, rn, name) in BAM_CONFIGS:
            log("\n" + "=" * 72)
            log(f"[{model_name}] {name}  (g1={g1} rACK={ra} rNACK={rn})"
                f"  — {N_TRIALS} trials")
            log(f"  confirmation: posterior matching at p={P_CONF} "
                f"(u_ACK=sym{SYM_ACK} angle0, u_NACK=sym{SYM_NACK} anglePi), "
                f"eps_ACK={EPS_CONF}, b_conf={_B_CONF:.4f}")
            log("=" * 72)
            tm = TrialMetrics()
            for i in range(N_TRIALS):
                prompt_ids = pool[i % len(pool)]
                m_true = int(np.random.RandomState(50000 + i).randint(M_MSG))
                np.random.seed(50000 + i)
                torch.manual_seed(50000 + i)
                ok, n, why, decoded, _stego_ids = burnashev_arcmark(
                    prompt_ids, m_true, MAX_TOKENS, g1, ra, rn)
                ber = bit_error_rate(decoded, m_true)
                tm.add(ok, n, ber, why)
                log(f"  trial {i + 1:>3}/{N_TRIALS}: m={m_true:>3} -> "
                    f"{'OK' if ok else 'WRONG':>5} n={n:>3} ({why}) "
                    f"ber={ber:.3f}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            tm.log_summary(model_name, name)
            rows.append(tm.row(model_name, name))
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
# Combined report
# ============================================================================
log("\n" + "=" * 72); log("COMBINED SUMMARY (all models)"); log("=" * 72)
log(f"\n{'Model':<24} {'Scheme':<16} "
    f"{'msg_err':>8} {'ber':>7} {'avg_tok':>8} {'std':>7} {'se':>6} "
    f"{'forced':>7}")
log("-" * 90)
last_model = None
for row in all_results:
    short = row["model"].split('/')[-1]
    if short != last_model:
        if last_model is not None:
            log("-" * 90)
        last_model = short
    log(f"{short:<24} {row['scheme']:<16} "
        f"{row['err']:>8.4f} {row['ber']:>7.4f} "
        f"{row['tok']:>8.2f} {row['std']:>7.2f} {row['tok_se']:>6.2f} "
        f"{row['forced_frac']:>7.3f}")

with open(OUT_CSV, "w") as f:
    f.write("model,scheme,msg_err_rate,bit_err_rate,"
            "avg_tokens,std_tokens,se_tokens,forced_frac\n")
    for row in all_results:
        f.write(f"{row['model']},{row['scheme']},"
                f"{row['err']:.6f},{row['ber']:.6f},"
                f"{row['tok']:.4f},{row['std']:.4f},{row['tok_se']:.4f},"
                f"{row['forced_frac']:.4f}\n")
log(f"\nWrote {OUT_CSV}")


# ── Plot: error rate vs avg tokens ──────────────────────────────────────────
models_in_order = [m.split('/')[-1] for m in MODEL_NAMES
                   if any(r["model"] == m for r in all_results)]
n_models = max(1, len(models_in_order))
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
axes = axes[0]

for ax, short in zip(axes, models_in_order):
    rows = [r for r in all_results if r["model"].split('/')[-1] == short]
    pts = sorted((r["tok"], r["err"], r["scheme"]) for r in rows)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s-",
                color="red", markersize=8, label="BAM")
        for tt, ee, nm in pts:
            ax.annotate(nm, (tt, ee), textcoords="offset points",
                        xytext=(7, 5), fontsize=8)
    ax.set_xlabel("Average tokens"); ax.set_ylabel("Error rate")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(short)

fig.suptitle(f"BAM error rate on C4 RealNews (N={N_TRIALS})")
plt.tight_layout(); plt.savefig(OUT_PLOT, dpi=140)
log(f"Wrote {OUT_PLOT}\nDone.")
