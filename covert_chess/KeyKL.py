"""
kl_security_c4.py — Information-theoretic security (Cachin 2000) of ArcMark's
optimal-transport emission, measured on C4 RealNews + Llama-3.1-8B.

CORRECTED QUANTITY (seed-marginalized, at fixed context)
--------------------------------------------------------
The Cachin security quantity is, at each position t with the FULL context held
fixed:

        D( E_{k}[ W(x | u, k, s) ]  ||  W(x | s) )

where s is the STATE (the base LM top-k distribution Ptilde_t, a deterministic
function of the fixed context) and the expectation is over the KEY prior.

The key at a position is k = (s_index, pi) = f(Seed, x_{t-h:t-1}). With the
context fixed, the ONLY randomness in k is the shared secret Seed. Drawing Seed
uniformly therefore INDUCES the key prior. So we marginalize over SEEDS:

    Qhat_t(x) = (1/Nseed) * sum_{j=1}^{Nseed} cond_{ s(Seed_j) }( x ; pi(Seed_j) )

CRUCIAL: each Seed_j yields a DIFFERENT permutation pi(Seed_j), hence a DIFFERENT
OT problem (different cost matrix) and a DIFFERENT coupling. We then extract the
SINGLE row at that seed's realized s_index. This is NOT the average of the R rows
of one coupling — that average is identically Ptilde_t by the OT column-marginal
constraint (KL == 0, uninformative). Averaging over seeds has no such identity,
so D(Qhat_t || Ptilde_t) is genuinely nonzero and IS the Cachin overhead induced
by the (deterministic, context-keyed) key mechanism.

The symbol u is NOT a function of the seed (it comes from the message/BAM belief),
so at a fixed position u is held fixed at a representative value and only
(s_index, pi) are marginalized over seeds — matching E_k[ W(x|u,k,s) ].

KL is computed TOP-k vs TOP-k (both restricted to the OT top-k support and
renormalized), so any nonzero value is pure key-marginal distortion, not the
top-k truncation gap.

Sequence generation: at each step we emit ONE token using a fixed operational
seed (SHARED_SEED) so the trajectory is a valid watermarked sequence and the
context distribution is realistic; the KL at that step is computed by
marginalizing over Nseed FRESH uniform seeds at the same fixed context.

Sweep:
  R     (OT key cardinality / rotation slices) in {4, 8, 16}
  Nseed (Monte-Carlo seed count)               — a few values, to show convergence
  GEN_TOKENS horizon  in {25, 50, 75, 100}     — stationarity check: does the
             per-position KL drift with depth into the run? Implemented for FREE
             by running the LONGEST horizon once and slicing (positions 0..24 of
             a 100-token walk are identical to a standalone 25-token walk: same
             prompt, same fixed symbol, same operational seed => same trajectory).

Output: per-position + cumulative KL vs position (one curve per R); an
Nseed-convergence panel; and a horizon panel showing cumulative KL at each
GEN_TOKENS cutoff plus mean-per-position KL vs horizon (flat => stationary).
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
from arcmark.sinkhorn import extract_conditional, solve_arcmark_ot, restrict_vocab
from arcmark.side_info import SideInfoMode, compute_key_si
from arcmark import geometry as _geom


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAMES = [
    "unsloth/Meta-Llama-3.1-8B",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False

# ── How much text to characterise ───────────────────────────────────────────
N_PROMPTS_EVAL = 3 if not SMOKE_TEST else 3     # prompts (independent runs)

# Horizon sweep. We WALK the longest horizon once per run and SLICE the shorter
# cutoffs from it (free: shared prefix trajectory). GEN_TOKENS_MAX drives the
# actual walk length; HORIZON_SWEEP are the reporting/plotting cutoffs.
HORIZON_SWEEP  = [50, 100, 150, 200] if not SMOKE_TEST else [4, 8]
GEN_TOKENS_MAX = max(HORIZON_SWEEP)
GEN_TOKENS     = GEN_TOKENS_MAX                  # positions scored per run

# ── Security sweep ──────────────────────────────────────────────────────────
R_SWEEP     = [4] if not SMOKE_TEST else [4]
NSEED_SWEEP = [10000, 50000, 100000] if not SMOKE_TEST else [8, 32]   # Monte-Carlo seeds
NSEED_MAIN  = NSEED_SWEEP[-1]                                  # curve uses largest

# Operational seed used to actually EMIT the walked sequence (fixed).
SHARED_SEED = 0xA12C

OUT_PLOT       = "kl_security_c4.png"
OUT_CONV_PLOT  = "kl_security_c4_convergence.png"
OUT_HORIZ_PLOT = "kl_security_c4_horizon.png"
OUT_CSV        = "kl_security_c4.csv"

# ── ArcMark core knobs ──────────────────────────────────────────────────────
P_FIELD            = 4
TOP_K              = 50
SINKHORN_REG       = 0.2
SINKHORN_MAX_ITER  = 4000
SINKHORN_STOP_THR  = 1e-4

# ── Batched-solver speedup switches ─────────────────────────────────────────
# USE_BATCHED: run all Nseed OT solves for a position as ONE batched log-domain
#   Sinkhorn on GPU (levers 2+3+4). Falls back to the per-seed path if False.
# BATCHED_MAX_ITER: cut from SINKHORN_MAX_ITER (lever 1). 500 is plenty for a
#   top-50 support; verify once against 4000 before trusting.
# VERIFY_BATCHED: if True, at the FIRST few positions of the FIRST run, compute
#   the seed-marginal KL by BOTH paths and assert agreement, then continue with
#   the batched path. This is the correctness gate before large Nseed.
USE_BATCHED        = True
BATCHED_MAX_ITER   = 500
BATCHED_STOP_THR   = 1e-4
VERIFY_BATCHED     = True
VERIFY_N_POSITIONS = 3        # positions to cross-check (first run only)
VERIFY_TOL         = 5e-4     # max |KL_batched - KL_perseed| tolerated (abs)
PHI                = 0.0

# Operational symbol u held fixed per position. The Cachin quantity conditions
# on u; the seed does not affect u. Any fixed u in {0,...,P_FIELD-1} is valid;
# the KL is measured conditional on it.
U_SYMBOL = 1

PROMPT_TOKEN_LEN = 32
N_PROMPTS_POOL   = 200

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

# R_RESOLUTION set per outer-sweep iteration.
R_RESOLUTION = R_SWEEP[0]


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
    log(f"Loading C4 RealNews prompts ({PROMPT_TOKEN_LEN} tokens each)...")
    pool: list[list[int]] = []
    try:
        ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        ds = ds.shuffle(seed=12345, buffer_size=2000)
        for ex in ds:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            if len(ids) >= PROMPT_TOKEN_LEN:
                pool.append(ids[:PROMPT_TOKEN_LEN])
            if len(pool) >= N_PROMPTS_POOL:
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
        for txt in fallbacks * (N_PROMPTS_POOL // len(fallbacks) + 1):
            ids = tokenizer.encode(txt, add_special_tokens=False)
            pool.append(
                ids[:PROMPT_TOKEN_LEN] if len(ids) >= PROMPT_TOKEN_LEN
                else ids + [tokenizer.eos_token_id] * (PROMPT_TOKEN_LEN - len(ids))
            )
            if len(pool) >= N_PROMPTS_POOL:
                break
    log(f"  loaded {len(pool)} prompts")
    return pool


# ============================================================================
# Key / permutation helpers
# ============================================================================
def _context_tokens_for_step(emitted: list[int], context_width: int) -> tuple[int, ...]:
    pad_len = max(0, context_width - len(emitted))
    return tuple([0] * pad_len + emitted[-context_width:])


def _perm_for_seed(perm_seed: int, device) -> torch.Tensor:
    cache = CTX.perm_cache
    perm = cache.get(perm_seed)
    if perm is None or perm.device != device:
        from arcmark import geometry
        perm = geometry.random_permutation(CTX.vocab_size, seed=perm_seed).to(device)
        # Many distinct seeds per position -> cap cache so it doesn't balloon.
        if len(cache) > 1024:
            cache.clear()
        cache[perm_seed] = perm
    return perm


# ============================================================================
# Core: seed-marginalized emission distribution + Cachin KL at a fixed context
# ============================================================================
@torch.no_grad()
def _cond_for_seed(probs: torch.Tensor, context_tokens: tuple[int, ...],
                   symbol: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For ONE seed at a FIXED context: derive (s_index, pi), solve the OT
    (permutation depends on seed -> distinct coupling), and return the SINGLE
    conditional row at that seed's realized s_index, on the top-k support.

    Returns (cond_on_topk[float64, shape (n_sel,)], token_indices).
    """
    s_index, perm_seed = compute_key_si(
        secret_key=seed,
        context_tokens=context_tokens,
        num_keys=R_RESOLUTION,
        mode=SIDE_INFO_MODE,
        tokenizer=CTX.tokenizer,
    )
    perm = _perm_for_seed(perm_seed, probs.device)
    ot = solve_arcmark_ot(
        probs,
        codeword_symbol=int(symbol),
        alphabet_size=P_FIELD,
        num_keys=R_RESOLUTION,
        vocab_size=CTX.vocab_size,
        perm=perm,
        phi=PHI,
        config=ARC_CONFIG,
    )
    # single row at the realized s_index, kept on the coupling's own top-k support
    cond = extract_conditional(
        ot.coupling, s_index,
        num_keys=R_RESOLUTION,
        full_vocab_size=None,
        token_indices=None,
    ).double()
    return cond, ot.token_indices


@torch.no_grad()
def seed_marginal_kl(probs: torch.Tensor, context_tokens: tuple[int, ...],
                     symbol: int, seeds: list[int]):
    """Cachin KL at a fixed context, marginalized over SEEDS.

    Qhat = (1/Nseed) sum_j cond_{s(Seed_j)}( . ; pi(Seed_j) ), accumulated on the
    FULL vocab then restricted to the (seed-independent) top-k support for the KL.
    Base = Ptilde = renormalized top-k base (the OT state s = W(x|s)).

    Returns (kl_full, ckpt_kls) where ckpt_kls maps each Nseed checkpoint in
    NSEED_SWEEP (<= len(seeds)) to the sub-sample KL, for the convergence panel.
    """
    V = CTX.vocab_size
    Qhat_full = torch.zeros(V, dtype=torch.float64, device=probs.device)
    common_idx = None
    checkpoints = sorted(set(n for n in NSEED_SWEEP if n <= len(seeds)))
    ckpt_kls: dict[int, float] = {}

    for j, seed in enumerate(seeds, start=1):
        cond, idx = _cond_for_seed(probs, context_tokens, symbol, seed)
        if common_idx is None:
            common_idx = idx
        Qhat_full.scatter_add_(0, idx.to(probs.device), cond)
        if j in checkpoints:
            ckpt_kls[j] = _kl_topk(Qhat_full / float(j), probs, common_idx)

    # KL at each Nseed checkpoint (the running seed-average evaluated at 16, 64,
    # 256, ...). The largest checkpoint is the main estimate. Returned as a dict
    # so every downstream table/plot can be produced per-Nseed for free — the
    # running average passes through the smaller Nseed on the way to the largest.
    return ckpt_kls


def _kl_topk(Qhat_full: torch.Tensor, probs: torch.Tensor,
             idx: torch.Tensor) -> float:
    """D(Qhat || Ptilde) on the top-k support `idx`, both renormalized."""
    P = probs.double()
    P_k = P[idx]; P_k = P_k / P_k.sum().clamp_min(1e-30)     # Ptilde
    Q_k = Qhat_full[idx]; Q_k = Q_k / Q_k.sum().clamp_min(1e-30)
    mask = Q_k > 0
    q = Q_k[mask]; p = P_k[mask].clamp_min(1e-30)
    kl = float(torch.sum(q * (torch.log(q.clamp_min(1e-30)) - torch.log(p))).item())
    return kl if math.isfinite(kl) else 0.0


# ============================================================================
# BATCHED seed-marginal KL  (levers 1-4)
# ----------------------------------------------------------------------------
# All Nseed OT solves at ONE position share the same source marginal
# (uniform 1/r), the same target marginal (top-k renormalized base, seed-
# independent), and the same target angles z^{(j)} (depend on symbol+rotation+
# phi, NOT the permutation). Only the TOKEN angles differ across seeds, because
# each seed's permutation relabels which top-k token sits where on the circle.
# So we build ONE (B, r, k) cost tensor and run ONE batched log-domain Sinkhorn.
#
# Per position:
#   1. restrict_vocab ONCE -> (idx[k], probs_sel[k])                 (lever 3b)
#   2. for each of B seeds: (s_index, perm_seed); gather perm ONLY at the k
#      top-k tokens -> permuted ids, -> token angles theta[B,k]      (lever 3a,4)
#   3. batched cost C[B,r,k] = circular_dist(z[r], theta[B,k])       (lever 4)
#   4. batched log-Sinkhorn -> coupling[B,r,k]                       (lever 2)
#   5. cond[b] = r * coupling[b, s_index[b], :]  (the realized row per seed)
#   6. Qhat = mean_b scatter(cond[b]) ; KL top-k vs top-k
# ============================================================================

def _sideinfo_angles(symbol: int, r: int, device) -> torch.Tensor:
    """z^{(j)} = (2*pi*symbol/p + 2*pi*j/r + phi) mod 2pi, shape (r,). Seed-
    independent. Mirrors geometry.side_info_angles."""
    base = _geom.TWO_PI * float(symbol) / float(P_FIELD)
    j = torch.arange(r, dtype=torch.float64, device=device)
    return (base + _geom.TWO_PI * j / float(r) + PHI) % _geom.TWO_PI


@torch.no_grad()
def _batched_log_sinkhorn(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                          reg: float, max_iter: int, stop_thr: float) -> torch.Tensor:
    """Batched entropic OT in the log domain. Mirrors POT's 'sinkhorn_log'.

    cost: (B, m, n)  a: (m,)  b: (n,)   -> coupling (B, m, n), rows sum to a,
    cols sum to b (to stop_thr). Stable log-sum-exp updates.
    """
    B, m, n = cost.shape
    Mr = -cost / reg                                   # (B,m,n)
    log_a = torch.log(a.clamp_min(1e-300)).view(1, m, 1)
    log_b = torch.log(b.clamp_min(1e-300)).view(1, 1, n)
    f = torch.zeros(B, m, 1, dtype=cost.dtype, device=cost.device)
    g = torch.zeros(B, 1, n, dtype=cost.dtype, device=cost.device)
    for it in range(max_iter):
        # f update: log_a - logsumexp_n (Mr + g)
        f = log_a - torch.logsumexp(Mr + g, dim=2, keepdim=True)
        g_new = log_b - torch.logsumexp(Mr + f, dim=1, keepdim=True)
        # convergence on the g marginal every few iters
        if it % 10 == 0 or it == max_iter - 1:
            err = (g_new - g).abs().max().item()
            g = g_new
            if err < stop_thr:
                break
        else:
            g = g_new
    coupling = torch.exp(Mr + f + g)                   # (B,m,n)
    return coupling


@torch.no_grad()
def _perm_topk_positions(perm_seed: int, idx: torch.Tensor) -> torch.Tensor:
    """Permuted vocab positions of ONLY the top-k tokens (lever 3a).

    We need perm[idx] (k entries), not the whole 128k permutation. We still
    build the full permutation once per seed (torch.randperm is O(V)) but this
    is unavoidable to match geometry.random_permutation EXACTLY; the win is we
    do NOT cache/scatter full-vocab tensors downstream. If exact match to the
    per-seed path were not required, a k-subset hash would be cheaper.
    """
    perm = _perm_for_seed(perm_seed, idx.device)       # (V,) cached
    return perm[idx]                                   # (k,)


@torch.no_grad()
def seed_marginal_kl_batched(probs: torch.Tensor, context_tokens: tuple[int, ...],
                             symbol: int, seeds: list[int]):
    """Batched Cachin KL at a fixed context, marginalized over SEEDS.
    Returns {Nseed_checkpoint: kl}, matching seed_marginal_kl's contract."""
    device = probs.device
    r = R_RESOLUTION

    # 1. top-k restriction ONCE (seed-independent)
    idx, probs_sel = restrict_vocab(
        probs, top_k=ARC_CONFIG.top_k, top_p=ARC_CONFIG.top_p,
        min_tokens=ARC_CONFIG.min_tokens,
    )
    k = idx.numel()
    b_marg = probs_sel.double()                        # (k,) target marginal
    a_marg = torch.full((r,), 1.0 / r, dtype=torch.float64, device=device)
    z = _sideinfo_angles(symbol, r, device)            # (r,)

    B = len(seeds)
    # 2. per-seed: s_index and token angles theta[B,k]
    s_index = torch.empty(B, dtype=torch.long, device=device)
    theta = torch.empty(B, k, dtype=torch.float64, device=device)
    for bi, seed in enumerate(seeds):
        s_idx, perm_seed = compute_key_si(
            secret_key=int(seed), context_tokens=context_tokens,
            num_keys=r, mode=SIDE_INFO_MODE, tokenizer=CTX.tokenizer,
        )
        s_index[bi] = int(s_idx)
        permuted = _perm_topk_positions(int(perm_seed), idx).double()   # (k,)
        theta[bi] = _geom.TWO_PI * permuted / float(CTX.vocab_size)

    # 3. batched cost C[B,r,k] = circular_dist(z[r,1], theta[B,1,k])
    #    broadcast: z -> (1,r,1), theta -> (B,1,k)
    diff = (z.view(1, r, 1) - theta.view(B, 1, k)) % _geom.TWO_PI
    cost = torch.minimum(diff, _geom.TWO_PI - diff)    # (B,r,k), in [0,pi]

    # 4. batched log-Sinkhorn
    coupling = _batched_log_sinkhorn(
        cost, a_marg, b_marg, reg=SINKHORN_REG,
        max_iter=BATCHED_MAX_ITER, stop_thr=BATCHED_STOP_THR,
    )                                                  # (B,r,k)

    # 5. realized row per seed: cond[b] = r * coupling[b, s_index[b], :]
    rows = coupling[torch.arange(B, device=device), s_index, :]  # (B,k)
    rows = rows * float(r)
    rows = rows.clamp_min(0.0)
    rows = rows / rows.sum(dim=1, keepdim=True).clamp_min(1e-300) # renorm each

    # 6. running seed-average -> KL at each checkpoint
    checkpoints = sorted(set(n for n in NSEED_SWEEP if n <= B))
    ckpt_kls: dict[int, float] = {}
    Qk_run = torch.zeros(k, dtype=torch.float64, device=device)
    # Ptilde on the top-k support (== b_marg already renormalized)
    P_k = b_marg / b_marg.sum().clamp_min(1e-30)
    logP = torch.log(P_k.clamp_min(1e-30))
    for j in range(1, B + 1):
        Qk_run += rows[j - 1]
        if j in checkpoints:
            Qk = Qk_run / float(j)
            Qk = Qk / Qk.sum().clamp_min(1e-30)
            m = Qk > 0
            kl = float(torch.sum(Qk[m] * (torch.log(Qk[m].clamp_min(1e-30)) - logP[m])).item())
            ckpt_kls[j] = kl if math.isfinite(kl) else 0.0
    return ckpt_kls


# ============================================================================
# Emission of the walked sequence (operational seed, fixed)
# ============================================================================
@torch.no_grad()
def emit_operational_token(probs: torch.Tensor, context_tokens: tuple[int, ...],
                           symbol: int) -> int:
    """Emit ONE token using the fixed operational SHARED_SEED, to advance the
    realistic watermarked trajectory (context distribution)."""
    cond, idx = _cond_for_seed(probs, context_tokens, symbol, SHARED_SEED)
    tok_local = int(torch.multinomial(cond.float(), num_samples=1).item())
    return int(idx[tok_local].item())


# ============================================================================
# One run: walk GEN_TOKENS positions; at each, seed-marginalized Cachin KL
# ============================================================================
# module-level flag: has the batched<->perseed verification already run?
_VERIFY_DONE = False


def _verify_batched_vs_perseed(probs, ctx, seeds, pos):
    """Compute the seed-marginal KL both ways at one position and check they
    agree to VERIFY_TOL. Raises AssertionError on mismatch."""
    ck_b = seed_marginal_kl_batched(probs, ctx, U_SYMBOL, seeds)
    ck_p = seed_marginal_kl(probs, ctx, U_SYMBOL, seeds)
    worst = 0.0
    for n in sorted(set(ck_b) & set(ck_p)):
        d = abs(ck_b[n] - ck_p[n])
        worst = max(worst, d)
        log(f"    [verify pos={pos} Nseed={n}] batched={ck_b[n]:.6e} "
            f"perseed={ck_p[n]:.6e} |Δ|={d:.2e}")
    if worst > VERIFY_TOL:
        raise AssertionError(
            f"Batched vs per-seed KL disagree by {worst:.2e} > tol {VERIFY_TOL:.2e} "
            f"at position {pos}. Tighten BATCHED_MAX_ITER/STOP_THR or investigate.")
    log(f"    [verify pos={pos}] OK (worst |Δ|={worst:.2e} <= {VERIFY_TOL:.0e})")
    return ck_b


def run_single(prompt_ids: list[int], seeds: list[int]):
    """Walk GEN_TOKENS positions; at each, get the seed-marginal Cachin KL at
    EVERY Nseed checkpoint. Returns a dict {Nseed: per_position_KL_array}.

    Uses the batched solver when USE_BATCHED; if VERIFY_BATCHED and this is the
    first run, cross-checks the first VERIFY_N_POSITIONS positions against the
    per-seed path before trusting the batched results.
    """
    global _VERIFY_DONE
    checkpoints = sorted(set(n for n in NSEED_SWEEP if n <= len(seeds)))
    per_pos_by_nseed: dict[int, list[float]] = {n: [] for n in checkpoints}

    lm = IncrementalLM(list(prompt_ids))
    emitted: list[int] = []
    try:
        for t in range(GEN_TOKENS):
            probs = lm.probs()
            ctx = _context_tokens_for_step(emitted, ARC_CONFIG.context_width)

            if USE_BATCHED:
                if (VERIFY_BATCHED and not _VERIFY_DONE
                        and t < VERIFY_N_POSITIONS):
                    ckpt = _verify_batched_vs_perseed(probs, ctx, seeds, t)
                    if t == VERIFY_N_POSITIONS - 1:
                        _VERIFY_DONE = True
                        log("    [verify] batched path confirmed; "
                            "continuing batched-only.")
                else:
                    ckpt = seed_marginal_kl_batched(probs, ctx, U_SYMBOL, seeds)
            else:
                ckpt = seed_marginal_kl(probs, ctx, U_SYMBOL, seeds)

            for n in checkpoints:
                per_pos_by_nseed[n].append(ckpt[n])

            x = emit_operational_token(probs, ctx, U_SYMBOL)
            emitted.append(x)
            lm.advance(x)
        return {n: np.array(v) for n, v in per_pos_by_nseed.items()}
    finally:
        lm.free()


# ============================================================================
# Sweep driver
# ============================================================================
def _draw_seeds(n: int, tag: int) -> list[int]:
    rng = np.random.RandomState(0xBEEF + tag)
    return [int(rng.randint(1, 2**62)) for _ in range(n)]  # uniform key prior


def _se(vals) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    m = vals.shape[0]
    if m <= 1:
        return 0.0
    return float(vals.std(ddof=1) / math.sqrt(m))


def _aggregate(runs: np.ndarray, R_value: int, n_seed: int) -> dict:
    """From a (n_runs, GEN_TOKENS) matrix of per-position KL, build all
    aggregates + run-level SEs (std across runs / sqrt(n_runs))."""
    n_runs = runs.shape[0]
    mean_by_pos = runs.mean(axis=0) if n_runs else np.zeros(GEN_TOKENS)
    sem_by_pos  = (runs.std(axis=0, ddof=1) / math.sqrt(n_runs)
                   if n_runs > 1 else np.zeros(GEN_TOKENS))
    cum_by_pos  = np.cumsum(mean_by_pos)

    horizon_cum  = {h: float(cum_by_pos[h - 1]) for h in HORIZON_SWEEP if h <= GEN_TOKENS}
    horizon_mean = {h: float(cum_by_pos[h - 1] / h) for h in HORIZON_SWEEP if h <= GEN_TOKENS}

    per_run_meanpos = runs.mean(axis=1) if runs.size else np.zeros(0)
    per_run_cumtot  = runs.sum(axis=1)  if runs.size else np.zeros(0)

    horizon_cum_se:  dict[int, float] = {}
    horizon_mean_se: dict[int, float] = {}
    for h in HORIZON_SWEEP:
        if h <= GEN_TOKENS and runs.size:
            prh = runs[:, :h].sum(axis=1)
            horizon_cum_se[h]  = _se(prh)
            horizon_mean_se[h] = _se(prh / h)

    return {
        "R": R_value, "n_seed": n_seed, "n_runs": n_runs,
        "mean_by_pos": mean_by_pos, "sem_by_pos": sem_by_pos,
        "cum_by_pos": cum_by_pos,
        "mean_per_pos_kl": float(mean_by_pos.mean()) if GEN_TOKENS else 0.0,
        "mean_per_pos_se": _se(per_run_meanpos),
        "cum_total": float(cum_by_pos[-1]) if GEN_TOKENS else 0.0,
        "cum_total_se": _se(per_run_cumtot),
        "horizon_cum": horizon_cum, "horizon_cum_se": horizon_cum_se,
        "horizon_mean": horizon_mean, "horizon_mean_se": horizon_mean_se,
    }


def run_R(R_value: int, pool: list[list[int]]):
    global R_RESOLUTION
    R_RESOLUTION = R_value
    CTX.perm_cache.clear()

    seeds_main = _draw_seeds(NSEED_MAIN, tag=R_value)
    checkpoints = sorted(set(n for n in NSEED_SWEEP if n <= NSEED_MAIN))

    # For each Nseed checkpoint, collect one per-position KL vector per run.
    per_run_by_nseed: dict[int, list[np.ndarray]] = {n: [] for n in checkpoints}

    for i in range(N_PROMPTS_EVAL):
        prompt_ids = pool[i % len(pool)]
        t0 = time.time()
        vecs = run_single(prompt_ids, seeds_main)   # {Nseed: per_pos_array}
        dt = time.time() - t0
        for n in checkpoints:
            per_run_by_nseed[n].append(vecs[n])
        main_arr = vecs[NSEED_MAIN]
        log(f"  [R={R_value}] run {i+1:>3}/{N_PROMPTS_EVAL}: "
            f"meanKL/pos={main_arr.mean():.4e}  cumKL={main_arr.sum():.4e} "
            f"(Nseed={NSEED_MAIN}) [{dt:.1f}s]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build a full result (with SE) for EVERY Nseed checkpoint.
    results_by_nseed: dict[int, dict] = {}
    for n in checkpoints:
        runs = (np.vstack(per_run_by_nseed[n]) if per_run_by_nseed[n]
                else np.zeros((0, GEN_TOKENS)))
        results_by_nseed[n] = _aggregate(runs, R_value, n)

    # Log a compact per-Nseed summary for this R.
    log(f"  --> [R={R_value}] per-Nseed summary (mean±SE over {N_PROMPTS_EVAL} runs):")
    for n in checkpoints:
        r = results_by_nseed[n]
        log(f"      Nseed={n:>4}: meanKL/pos={r['mean_per_pos_kl']:.4e}"
            f"±{r['mean_per_pos_se']:.2e}  "
            f"cumKL={r['cum_total']:.4e}±{r['cum_total_se']:.2e}")
    return results_by_nseed


def run_model(model_name: str):
    global CTX
    CTX = LMContext(model_name)
    all_res: list[dict] = []       # flat list of per-(R,Nseed) result dicts
    try:
        CTX.prompt_pool = build_prompt_pool()
        pool = CTX.prompt_pool
        for R_value in R_SWEEP:
            log("\n" + "#" * 72)
            log(f"# R = {R_value}  (OT key cardinality); Nseed(main)={NSEED_MAIN}; "
                f"walk={GEN_TOKENS_MAX}, horizons={HORIZON_SWEEP}, "
                f"Nseed_sweep={NSEED_SWEEP}")
            log("#" * 72)
            results_by_nseed = run_R(R_value, pool)   # {Nseed: result_dict}
            for n in sorted(results_by_nseed):
                all_res.append(results_by_nseed[n])
    finally:
        CTX.teardown()
        CTX = None
    return all_res


# ============================================================================
# Run
# ============================================================================
all_results: list[dict] = []      # each dict is one (R, Nseed) result with SE
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
# Report + CSV
# ============================================================================
log("\n" + "=" * 72)
log("SEED-MARGINALIZED CACHIN KL  (D(E_seed[cond] || Ptilde), top-k vs top-k)")
log("=" * 72)
log(f"(SE = std across runs / sqrt(n_runs); n_runs={N_PROMPTS_EVAL})")
log(f"\n{'R':>4} {'Nseed':>6} {'meanKL/pos':>13} {'±SE':>11} "
    f"{'cumKL_total':>13} {'±SE':>11}")
log("-" * 66)
for res in sorted(all_results, key=lambda r: (r["R"], r["n_seed"])):
    log(f"{res['R']:>4} {res['n_seed']:>6} "
        f"{res['mean_per_pos_kl']:>13.4e} {res['mean_per_pos_se']:>11.3e} "
        f"{res['cum_total']:>13.4e} {res['cum_total_se']:>11.3e}")

log("\nHORIZON SWEEP (does length change anything?)  values as mean±SE")
log(f"{'R':>4} {'Nseed':>6}  " + "  ".join(f"cumH{h:<11}" for h in HORIZON_SWEEP)
    + " |  " + "  ".join(f"meanH{h:<11}" for h in HORIZON_SWEEP))
log("-" * 150)
for res in sorted(all_results, key=lambda r: (r["R"], r["n_seed"])):
    hcs = res["horizon_cum_se"]; hms = res["horizon_mean_se"]
    cum_str  = "  ".join(f"{res['horizon_cum'].get(h, float('nan')):.2e}±{hcs.get(h, 0.0):.1e}"
                         for h in HORIZON_SWEEP)
    mean_str = "  ".join(f"{res['horizon_mean'].get(h, float('nan')):.2e}±{hms.get(h, 0.0):.1e}"
                         for h in HORIZON_SWEEP)
    log(f"{res['R']:>4} {res['n_seed']:>6}  {cum_str} |  {mean_str}")

with open(OUT_CSV, "w") as f:
    f.write("R,n_seed,position,mean_kl,sem_kl,cum_kl\n")
    for res in sorted(all_results, key=lambda r: (r["R"], r["n_seed"])):
        for t in range(GEN_TOKENS):
            f.write(f"{res['R']},{res['n_seed']},{t},{res['mean_by_pos'][t]:.8e},"
                    f"{res['sem_by_pos'][t]:.8e},{res['cum_by_pos'][t]:.8e}\n")
    f.write("\n# horizon sweep (SE = std across runs / sqrt(n_runs))\n")
    f.write("R,n_seed,horizon,cum_kl,cum_kl_se,mean_per_pos_kl,mean_per_pos_kl_se,n_runs\n")
    for res in sorted(all_results, key=lambda r: (r["R"], r["n_seed"])):
        hcs = res["horizon_cum_se"]; hms = res["horizon_mean_se"]
        for h in HORIZON_SWEEP:
            if h in res["horizon_cum"]:
                f.write(f"{res['R']},{res['n_seed']},{h},{res['horizon_cum'][h]:.8e},"
                        f"{hcs.get(h, 0.0):.8e},{res['horizon_mean'][h]:.8e},"
                        f"{hms.get(h, 0.0):.8e},{res['n_runs']}\n")
    f.write("\n# aggregate scalars with SE\n")
    f.write("R,n_seed,mean_per_pos_kl,mean_per_pos_kl_se,cum_total_kl,cum_total_kl_se,n_runs\n")
    for res in sorted(all_results, key=lambda r: (r["R"], r["n_seed"])):
        f.write(f"{res['R']},{res['n_seed']},{res['mean_per_pos_kl']:.8e},"
                f"{res['mean_per_pos_se']:.8e},{res['cum_total']:.8e},"
                f"{res['cum_total_se']:.8e},{res['n_runs']}\n")
log(f"\nWrote {OUT_CSV}")


# ============================================================================
# Plot 1: cumulative + per-position KL vs position, one curve per R
# ============================================================================
colors = {4: "tab:blue", 8: "tab:orange", 16: "tab:green"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

main_results = [r for r in all_results if r["n_seed"] == NSEED_MAIN]
for res in sorted(main_results, key=lambda r: r["R"]):
    R = res["R"]
    x = np.arange(1, GEN_TOKENS + 1)
    ax1.plot(x, res["cum_by_pos"], marker="o", markersize=3,
             color=colors.get(R), label=f"R={R}")
    ax2.errorbar(x, res["mean_by_pos"], yerr=res["sem_by_pos"],
                 marker="o", markersize=3, capsize=2,
                 color=colors.get(R), label=f"R={R}")

# mark the horizon cutoffs on the cumulative panel
for h in HORIZON_SWEEP:
    if h <= GEN_TOKENS:
        ax1.axvline(h, color="gray", ls=":", lw=0.8, alpha=0.6)

ax1.set_xlabel("Position t")
ax1.set_ylabel(r"Cumulative $\sum_{\tau\le t} D(\hat{Q}_\tau \,\|\, \tilde{P}_\tau)$ (nats)")
ax1.set_title("Cachin KL accumulation (seed-marginalized)")
ax1.grid(True, alpha=0.3); ax1.legend(title="OT key card. R")

ax2.set_xlabel("Position t")
ax2.set_ylabel(r"Per-position $D(\hat{Q}_t \,\|\, \tilde{P}_t)$ (nats)")
ax2.set_title("Per-position Cachin KL")
ax2.grid(True, alpha=0.3); ax2.legend(title="OT key card. R")

fig.suptitle("Information-theoretic (Cachin) overhead of context-keyed OT\n"
             f"seed-marginalized, top-k vs top-k, C4 RealNews / Llama-3.1-8B "
             f"(Nseed={NSEED_MAIN}, {N_PROMPTS_EVAL} runs)")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=140)
log(f"Wrote {OUT_PLOT}")


# ============================================================================
# Plot 2: Nseed convergence — mean per-position KL vs Nseed, one line per R.
# Uses the FULL per-position KL at each Nseed (not just position 0), so the
# convergence is of the whole curve's mean, with proper run-level SE bars.
# ============================================================================
fig2, ax = plt.subplots(figsize=(7, 5))
for R in R_SWEEP:
    rs = sorted([r for r in all_results if r["R"] == R], key=lambda r: r["n_seed"])
    xs = [r["n_seed"] for r in rs]
    ys = [r["mean_per_pos_kl"] for r in rs]
    es = [r["mean_per_pos_se"] for r in rs]
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3,
                    color=colors.get(R), label=f"R={R}")
ax.set_xscale("log", base=2)
ax.set_xlabel(r"$N_{\mathrm{seed}}$ (Monte-Carlo seeds)")
ax.set_ylabel(r"mean per-position $D(\hat{Q}_t \,\|\, \tilde{P}_t)$ (nats)")
ax.set_title("Seed-marginal KL vs Monte-Carlo budget (whole-sequence mean)\n"
             "flattening => estimate converged; residual = true Cachin overhead")
ax.grid(True, alpha=0.3, which="both"); ax.legend(title="OT key card. R")
plt.tight_layout()
plt.savefig(OUT_CONV_PLOT, dpi=140)
log(f"Wrote {OUT_CONV_PLOT}")


# ============================================================================
# Plot 3: horizon sweep — does length change anything? (at NSEED_MAIN)
# ============================================================================
fig3, (axh1, axh2) = plt.subplots(1, 2, figsize=(13, 5))
for res in sorted(main_results, key=lambda r: r["R"]):
    R = res["R"]
    hs   = [h for h in HORIZON_SWEEP if h in res["horizon_cum"]]
    cums = [res["horizon_cum"][h]  for h in hs]
    mns  = [res["horizon_mean"][h] for h in hs]
    cums_se = [res["horizon_cum_se"].get(h, 0.0)  for h in hs]
    mns_se  = [res["horizon_mean_se"].get(h, 0.0) for h in hs]
    axh1.errorbar(hs, cums, yerr=cums_se, marker="o", capsize=3,
                  color=colors.get(R), label=f"R={R}")
    axh2.errorbar(hs, mns,  yerr=mns_se,  marker="o", capsize=3,
                  color=colors.get(R), label=f"R={R}")

axh1.set_xlabel("Horizon (GEN_TOKENS)")
axh1.set_ylabel(r"Cumulative KL over first $H$ positions (nats)")
axh1.set_title("Cumulative Cachin KL vs horizon\n(linear in H => stationary per-position KL)")
axh1.grid(True, alpha=0.3); axh1.legend(title="R"); axh1.set_xticks(HORIZON_SWEEP)

axh2.set_xlabel("Horizon (GEN_TOKENS)")
axh2.set_ylabel(r"Mean per-position KL over first $H$ positions (nats)")
axh2.set_title("Mean per-position Cachin KL vs horizon\n(flat => length changes nothing)")
axh2.grid(True, alpha=0.3); axh2.legend(title="R"); axh2.set_xticks(HORIZON_SWEEP)

fig3.suptitle(f"Horizon sensitivity (Nseed={NSEED_MAIN}, {N_PROMPTS_EVAL} runs)")
plt.tight_layout()
plt.savefig(OUT_HORIZ_PLOT, dpi=140)
log(f"Wrote {OUT_HORIZ_PLOT}\nDone.")