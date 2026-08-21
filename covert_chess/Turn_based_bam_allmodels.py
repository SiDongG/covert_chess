from __future__ import annotations


import math
import os
import sys
import time
import gc
import itertools

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

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
    'meta-llama/Llama-3.1-8B-Instruct',
    'microsoft/phi-4',                     # Phi-4, 14B dense, ungated (MIT)
    'Qwen/Qwen3-30B-A3B-Instruct-2507',    # 30B MoE (3B active), non-thinking
]
# Short filename tags per model (per-model CSV/PNG/rollout outputs).
MODEL_TAGS = {
    'meta-llama/Llama-3.1-8B-Instruct': 'llama8b',
    'microsoft/phi-4': 'phi4-14b',
    'Qwen/Qwen3-30B-A3B-Instruct-2507': 'qwen30b',
}
# Per-model device_map. Small models sit on one device; the 30B MoE
# (~60GB fp16) is sharded with "auto" (override via env DEVICE_MAP).
MODEL_DEVICE_MAP = {
    'Qwen/Qwen3-30B-A3B-Instruct-2507': os.environ.get("DEVICE_MAP", "auto"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise SystemExit(
        "No CUDA device visible — run inside a GPU allocation "
        "(check `echo $CUDA_VISIBLE_DEVICES` and `nvidia-smi`)."
    )

SMOKE_TEST = False
N_DIALOGUES = 100 if not SMOKE_TEST else 2
ROUNDS_PER_DIALOGUE = 10 if not SMOKE_TEST else 4

# Single BAM operating point.
BAM_G1   = 0.5
BAM_G2   = 0.9999999
BAM_RACK = 0.984375
BAM_RNACK = 0.75

MAX_CONF_STEPS  = 80
MIN_COMM_TOKENS = 0
# Total token budget for a single payload, summed across the turns it spans.
MAX_PAYLOAD_TOKENS = 800
# Confidence-gated spill: at a turn's EOS, if the relevant belief is below
# SPILL_BELIEF_THR (top-message belief, or rho_ACK while confirming), the
# payload does NOT force-decode — it spills to a filler/resume turn so it can
# finish later. After MAX_SPILLS spills it force-decodes regardless.
SPILL_BELIEF_THR = 0.75
MAX_SPILLS       = 3
N_PRINT_TRANSCRIPTS = 2

OUT_PLOT = "turnbased_bam_matrix.png"
OUT_CSV  = "turnbased_bam_matrix.csv"

# ── ArcMark core knobs (unchanged) ──────────────────────────────────────────
P_FIELD            = 4
R_RESOLUTION       = 4
SHARED_SEED        = 0x9E3779B97F4A7C15F39CC0605CEDC834  # 128-bit (lambda=128); matches compare.py
TOP_K              = 50
SINKHORN_REG       = 0.2
SINKHORN_MAX_ITER  = 4000
SINKHORN_STOP_THR  = 1e-4
PHI                = 0.0

# M_MSG is now PER-TASK (see TASKS). The reserved STOP payload is always the
# top index of whatever message space the task uses; here we keep a single
# global STOP convention by reserving the LAST index of each task's space.
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



import chess as _chess


class TaskGen:
    """Stateful generator. next_payload() returns (payload, M_MSG, coded_bits,
    label) where coded_bits = log2(num legal moves) for THIS move and label is
    a human-readable move string (e.g. 'e2e4'). Reserves index M_MSG-1 as STOP."""
    name = "base"
    def reset(self, rng): ...
    def next_payload(self, rng): ...
    def done(self): return False


class TicTacToeGen(TaskGen):
    name = "tictactoe"
    def reset(self, rng):
        self.b = [0]*9; self.p = 1
    def _legal(self):
        return [i for i in range(9) if self.b[i] == 0]
    def _winner(self):
        L=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a,c,d in L:
            if self.b[a]!=0 and self.b[a]==self.b[c]==self.b[d]:
                return self.b[a]
        return 0
    def next_payload(self, rng):
        if not self._legal() or self._winner()!=0:
            self.reset(rng)
        lm = self._legal()
        n = len(lm)
        idx = int(rng.randint(n))            # index into legal set
        move = lm[idx]
        self.b[move] = self.p; self.p = 3 - self.p
        M = n + 1                            # +1 reserved STOP slot
        r, c = divmod(move, 3)               # 0-indexed row, col on 3x3 board
        label = f"({r+1},{c+1})"             # 1-indexed (row,col)
        return idx, M, math.log2(n), label


class ChessGen(TaskGen):
    """Two agents exchange chess moves. The payload each turn is the index of
    a uniformly chosen legal move; coded bits = log2(#legal moves), which sits
    around 5-6 bits in typical positions (≈20-40 legal moves)."""
    name = "chess"
    def reset(self, rng):
        self.bd = _chess.Board()
    def next_payload(self, rng):
        lm = list(self.bd.legal_moves)
        if not lm:
            self.reset(rng); lm = list(self.bd.legal_moves)
        n = len(lm)
        idx = int(rng.randint(n))
        label = lm[idx].uci()                # e.g. "e2e4", "g1f3"
        self.bd.push(lm[idx])
        M = n + 1                            # +1 reserved STOP slot
        return idx, M, math.log2(n), label


class Go19Gen(TaskGen):
    name = "go19"
    N = 19
    # Go column letters skip 'I' by convention: A-H, J-T for a 19-wide board.
    _COLS = "ABCDEFGHJKLMNOPQRST"
    def reset(self, rng):
        self.occ = set()
    def next_payload(self, rng):
        pts = self.N * self.N
        free = pts - len(self.occ)
        if free <= 1:
            self.reset(rng); free = pts
        # legal = empty points (ko/suicide ignored for the entropy channel;
        # the payload is the index into the empty-point list)
        empties = [x for x in range(pts) if x not in self.occ]
        n = len(empties)
        idx = int(rng.randint(n))
        point = empties[idx]
        self.occ.add(point)
        M = n + 1
        row, col = divmod(point, self.N)     # 0-indexed
        # Go coords: column letter + row number (1 at bottom -> use N-row)
        label = f"{self._COLS[col]}{self.N - row}"
        return idx, M, math.log2(n), label


TASKS = {
    "tictactoe": TicTacToeGen,
    "chess":     ChessGen,
    "go19":      Go19Gen,
}


# ============================================================================
# CONVERSATION PROFILES — per-turn token budgets + framing.
#
# Topic/opener SEEDS are loaded from an external dataset file
# (SEEDS_PATH, default conversation_seeds.json) with 50 {topic, opener} pairs
# per setting. Each dialogue samples one seed, so cover text varies across
# dialogues (topic-robustness + more independent samples).
#
# The side_a / side_b prompts here are TOPIC-AGNOSTIC role selectors: they name
# which speaker the agent is and the register to write in, but say nothing
# about the subject or stance. All specifics — including, for debates, BOTH
# opposing positions — live in the seed's `topic` string.
# ============================================================================
import json as _json

SEEDS_PATH = os.environ.get(
    "SEEDS_PATH",
    os.path.join(_THIS_DIR, "conversation_seeds.json"),
)


def _load_seed_banks(path):
    """Load {setting: [{topic, opener}, ...]} from JSON. Fails loudly if the
    file is missing so a run never silently falls back to a single topic."""
    if not os.path.exists(path):
        raise SystemExit(
            f"Seed dataset not found at {path}. Generate it with "
            f"make_seeds.py, or set SEEDS_PATH to its location."
        )
    with open(path) as f:
        banks = _json.load(f)
    for setting, seeds in banks.items():
        if not seeds:
            raise SystemExit(f"Seed bank '{setting}' is empty in {path}.")
        for s in seeds:
            if "topic" not in s or "opener" not in s:
                raise SystemExit(
                    f"Malformed seed in '{setting}': needs 'topic' and 'opener'.")
    return banks


SEED_BANKS = _load_seed_banks(SEEDS_PATH)
log(f"Loaded seed banks from {SEEDS_PATH}: "
    + ", ".join(f"{k}={len(v)}" for k, v in SEED_BANKS.items()))


class ConvProfile:
    name = "base"
    max_turn_tokens = 300
    max_filler_tokens = 64
    side_a = ""
    side_b = ""

    @property
    def seeds(self):
        return SEED_BANKS.get(self.name, [])


class ChatProfile(ConvProfile):
    name = "chat"
    max_turn_tokens = 90          # middle ground between terse and rambly
    max_filler_tokens = 40
    side_a = ("You are Friend A in a casual text chat with a friend. Reply "
              "naturally, like a real text message — lowercase, conversational. "
              "Reply should be within 50 words.")
    side_b = ("You are Friend B in a casual text chat with a friend. Reply "
              "naturally, like a real text message — lowercase, conversational. "
              "Reply should be within 50 words.")


class DebateProfile(ConvProfile):
    name = "debate"
    max_turn_tokens = 300         # medium, EOS-stable
    max_filler_tokens = 64
    side_a = ("You are User A in a Reddit-style debate thread. Argue your "
              "assigned position (given in the setup) naturally, staying in "
              "character and defending your side. Reply to the most recent "
              "message, like a Reddit exchange. Reply should be within 100 words.")
    side_b = ("You are User B in a Reddit-style debate thread. Argue your "
              "assigned position (given in the setup) naturally, staying in "
              "character and defending your side. Reply to the most recent "
              "message, like a Reddit exchange. Reply should be within 100 words.")


CONVERSATIONS = {
    "chat":    ChatProfile(),
    "debate":  DebateProfile(),
}


def pick_seed(profile, dialogue_index):
    """Deterministically choose a {topic, opener} seed for a dialogue. Cycles
    through the bank by index so runs are reproducible and every seed is used
    equally (with 100 dialogues and 50 seeds, each seed is used exactly twice)."""
    bank = profile.seeds
    if not bank:
        return {"topic": "", "opener": ""}
    return bank[dialogue_index % len(bank)]



# ============================================================================
# Per-model context  (unchanged except prompts now come from the profile)
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
            model_name, torch_dtype=torch.float16,
            device_map=MODEL_DEVICE_MAP.get(model_name, DEVICE),
        )
        self.model.eval()
        self.vocab_size = self.model.get_output_embeddings().weight.shape[0]
        if self.vocab_size != self.model.config.vocab_size:
            log(f"  note: lm_head width {self.vocab_size} != "
                f"config.vocab_size {self.model.config.vocab_size}; using lm_head.")
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        self.eos_ids = self._collect_turn_end_ids()
        log(f"  turn-end token ids: {sorted(self.eos_ids)}")
        self.boundary_ids = self._collect_boundary_ids()
        log(f"  boundary/role token ids (masked from carriers): "
            f"{sorted(self.boundary_ids)}")
        self.perm_cache: dict[int, torch.Tensor] = {}
        log(f"Loaded {model_name}. vocab_size={self.vocab_size}")

    def _collect_boundary_ids(self) -> set[int]:
        """Tokens that must never be emitted as watermark carriers because they
        are chat-template turn/role scaffolding. If steering emits these, the
        decoded text shows seams like 'assistant' and run-on multi-turn replies.
        Includes all EOS/turn-end ids plus role-header tokens."""
        ids = set(self.eos_ids)
        tok = self.tokenizer
        for s in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                  "<|im_start|>", "<|im_end|>", "<|end|>", "<|eom_id|>",
                  "<|python_tag|>", "assistant", "user", "system",
                  "<|start_header_id|>assistant<|end_header_id|>"):
            try:
                tid = tok.convert_tokens_to_ids(s)
            except Exception:
                tid = None
            if tid is not None and tid >= 0 and tid != tok.unk_token_id:
                ids.add(int(tid))
        return ids

    def _collect_turn_end_ids(self) -> set[int]:
        ids: set[int] = set()
        tok = self.tokenizer
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        for s in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>"):
            try:
                tid = tok.convert_tokens_to_ids(s)
            except Exception:
                tid = None
            if tid is not None and tid >= 0 and tid != tok.unk_token_id:
                ids.add(int(tid))
        gc_cfg = getattr(self.model, "generation_config", None)
        if gc_cfg is not None and getattr(gc_cfg, "eos_token_id", None) is not None:
            e = gc_cfg.eos_token_id
            if isinstance(e, (list, tuple)):
                ids.update(int(x) for x in e)
            else:
                ids.add(int(e))
        return ids

    def render_dialogue(self, history, speaking_agent, profile, topic) -> list[int]:
        side = profile.side_a if speaking_agent == 0 else profile.side_b
        sys_prompt = f"{topic}\n\n{side}"
        msgs = [{"role": "system", "content": sys_prompt}]
        for turn in history:
            role = "assistant" if turn["agent"] == speaking_agent else "user"
            msgs.append({"role": role, "content": turn["content"]})
        text = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def teardown(self):
        self.model = None
        self.tokenizer = None
        self.perm_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


CTX: "LMContext | None" = None


class IncrementalLM:
    def __init__(self, prompt_ids):
        model = CTX.model
        ids = torch.tensor(prompt_ids, dtype=torch.long,
                           device=model.device).unsqueeze(0)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        self.past = out.past_key_values
        self._last_logits = out.logits[0, -1].float()

    @torch.no_grad()
    def probs(self):
        return torch.softmax(self._last_logits, dim=-1)

    @torch.no_grad()
    def advance(self, token_id):
        model = CTX.model
        ids = torch.tensor([[token_id]], dtype=torch.long, device=model.device)
        out = model(ids, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self._last_logits = out.logits[0, -1].float()

    def free(self):
        self.past = None
        self._last_logits = None


# ============================================================================
# Shared emission / read primitives  (unchanged)
# ============================================================================
def _context_tokens_for_step(emitted, context_width):
    pad_len = max(0, context_width - len(emitted))
    return tuple([0] * pad_len + emitted[-context_width:])


def _perm_for_seed(perm_seed, device):
    cache = CTX.perm_cache
    perm = cache.get(perm_seed)
    if perm is None or perm.device != device:
        from arcmark import geometry
        perm = geometry.random_permutation(CTX.vocab_size, seed=perm_seed).to(device)
        if len(cache) > 256:
            cache.clear()
        cache[perm_seed] = perm
    return perm


def side_info_for_step(key_context) -> tuple[int, int, float]:
    """Synchronized per-token side information (s_index, perm_seed, R_t).

    One keyed SHA-256 over (secret || context) split into three disjoint
    blocks: channel index k_t^{(1)} (s_index), permutation seed Lambda_t^{(2)}
    (perm_seed), and posterior-matching randomness R_t = Rand(Lambda_t^{(3)}).
    R_t is derived from the shared seed and shared transcript, so encoder and
    decoder reconstruct it identically — it is NOT a local np.random draw.
    """
    s_index, perm_seed, R_t = compute_key_si(
        secret_key=SHARED_SEED, context_tokens=key_context,
        num_keys=R_RESOLUTION, mode=SIDE_INFO_MODE, tokenizer=CTX.tokenizer,
        return_r=True,
    )
    return s_index, perm_seed, R_t


@torch.no_grad()
def emit_token(probs, key_context, symbol, alphabet_size=P_FIELD):
    """Sample a token embedding ``symbol`` from an alphabet of size
    ``alphabet_size`` (p). ``alphabet_size`` is a parameter so the confirmation
    phase can emit through a genuine p=2 antipodal channel (paper: 'apply
    Algorithm 1 with p=2')."""
    s_index, perm_seed, _ = side_info_for_step(key_context)
    perm = _perm_for_seed(perm_seed, probs.device)
    ot_result = solve_arcmark_ot(
        probs, codeword_symbol=int(symbol), alphabet_size=int(alphabet_size),
        num_keys=R_RESOLUTION, vocab_size=CTX.vocab_size, perm=perm,
        phi=PHI, config=ARC_CONFIG,
    )
    cond = extract_conditional(
        ot_result.coupling, s_index, num_keys=R_RESOLUTION,
        full_vocab_size=CTX.vocab_size, token_indices=ot_result.token_indices,
    )
    token = int(torch.multinomial(cond, num_samples=1).item())
    base_logprob = float(torch.log(probs[token].clamp_min(1e-30)).item())
    return token, base_logprob


def read_symbol_angle(token_id, key_context):
    s_index, perm_seed, _ = side_info_for_step(key_context)
    perm = _perm_for_seed(perm_seed, CTX.model.device)
    permuted_id = int(perm[token_id].item())
    theta = (2.0 * math.pi) * permuted_id / float(CTX.vocab_size)
    s_angle = (2.0 * math.pi) * s_index / float(R_RESOLUTION)
    return (theta - s_angle) % (2.0 * math.pi)


# ============================================================================
# BAM posterior machinery — robust LAPLACE  (unchanged math; M is now passed in)
# ============================================================================
P_SYM      = P_FIELD
_SIGMA     = math.pi / P_SYM
_B_LAPLACE = float(os.environ.get("LAPLACE_B", _SIGMA / math.sqrt(2.0)))
EPS_NOISE  = 0.7
_Z_SIGNAL  = 2.0 * _B_LAPLACE * (1.0 - math.exp(-math.pi / _B_LAPLACE))


def posterior_match_symbol(pi, m, R):
    V = float(pi[:m].sum() + R * pi[m])
    return min(int(P_SYM * V), P_SYM - 1)


def per_symbol_likelihood(angle_obs):
    signal = np.empty(P_SYM)
    for u in range(P_SYM):
        target = (2 * math.pi * u / P_SYM + PHI) % (2 * math.pi)
        d = abs(angle_obs - target) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        signal[u] = math.exp(-d / _B_LAPLACE) / _Z_SIGNAL
    return (1.0 - EPS_NOISE) * signal + EPS_NOISE / (2.0 * math.pi)


def message_likelihood(ells, pi):
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
# CONFIRMATION phase — Algorithm 1 (posterior matching) at p = 2
# ============================================================================
# Faithful p=2 instantiation: antipodal symbols u_ACK = 0 (angle 0) and
# u_NACK = 1 (angle pi at p = 2), emitted through a genuine p = 2 ArcMark OT
# channel, with belief updated by the SAME contaminated-Laplace form at p = 2.
# This file keeps its OWN confirmation parameters (EPS_CONF = 0.3 and
# _B_CONF via _SIGMA_CONF = pi/2, i.e. b = pi/(2*sqrt(2)) — which already equals
# the p=2 scale pi/(P_CONF*sqrt(2))); only the emission alphabet and symbol
# mapping change to the faithful p = 2 form.
P_CONF   = 2
SYM_ACK  = 0                     # u_ACK  -> angle 0
SYM_NACK = 1                     # u_NACK -> angle pi at p = 2

EPS_CONF   = 0.3
_SIGMA_CONF = math.pi / 2.0
_B_CONF     = float(os.environ.get("LAPLACE_B_CONF", _SIGMA_CONF / math.sqrt(2.0)))
_Z_CONF     = 2.0 * _B_CONF * (1.0 - math.exp(-math.pi / _B_CONF))


def _circ_dist(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def conf_symbol_likelihood(angle_obs):
    """Contaminated-Laplace per-symbol likelihood at p = 2 over {u_ACK, u_NACK}.

    Same functional form as per_symbol_likelihood, instantiated at p = P_CONF
    = 2 with this file's confirmation Laplace scale _B_CONF and floor EPS_CONF.
    Returns [ell(u_ACK), ell(u_NACK)].
    """
    signal = np.empty(P_CONF)
    for u in range(P_CONF):
        target = (2 * math.pi * u / P_CONF + PHI) % (2 * math.pi)
        d = _circ_dist(angle_obs, target)
        signal[u] = math.exp(-d / _B_CONF) / _Z_CONF
    return (1.0 - EPS_CONF) * signal + EPS_CONF / (2.0 * math.pi)


# ============================================================================
# Payload transmission state — now carries M_MSG and coded_bits for THIS payload
# ============================================================================
class PayloadState:
    def __init__(self, payload, M_msg, coded_bits, label=""):
        self.payload = payload
        self.M = M_msg
        self.stop_msg = M_msg - 1            # reserved STOP slot for this task
        self.coded_bits = coded_bits
        self.label = label                   # human-readable move string
        self.pi = np.ones(M_msg) / M_msg
        self.knockdown = np.ones(M_msg)
        self.tokens_used = 0
        self.in_progress = True
        self.in_confirmation = False
        self.conf_rho = np.array([0.5, 0.5])
        self.conf_cand = -1
        self.conf_true_bit = 0
        self.n_spills = 0                    # times this payload has spilled


def dist_entropy_bits(probs):
    """Shannon entropy (bits) of a torch prob vector — the unwatermarked
    next-token distribution from the base LM, before ArcMark steering."""
    p = probs.double()
    nz = p > 0
    return float(-(p[nz] * torch.log2(p[nz])).sum().item())


class TurnResult:
    def __init__(self):
        self.tokens = []
        self.n_tokens = 0
        self.n_pad_tokens = 0
        self.base_logprobs = []
        self.entropies = []          # per-step entropy (bits) of unwm dist
        self.completed = False
        self.ended_by_eos = False
        self.decoder_decoded = -1
        self.outcome = ""
        self.mis_ack = False


# ============================================================================
# Watermarked turn — M_MSG now taken from the PayloadState (st.M). The
# per-turn token cap is passed in from the conversation profile.
# ============================================================================
def run_watermarked_turn(prompt_ids, history_tokens, st, max_turn_tokens,
                         g1, g2, ra, rn) -> TurnResult:
    """EOS-as-sole-decode-trigger model.

    The turn always generates watermarked tokens until the base model samples
    an EOS (or a token cap is hit), and only THEN decodes, taking argmax of the
    effective posterior at that point. Threshold crossings (g1/g2) and the
    ACK/NACK confirmation phase still run and update belief, but they NO LONGER
    end the turn — they are decorative under this scheme. Consequences:
      * every payload resolves at its turn's EOS  -> 100% completion,
      * one payload per turn (no spill / resume / multi-turn payloads),
      * no post-payload padding phase (the turn already ran to EOS), so a
        completed turn's carrier/total token ratio is 100%.
    """
    res = TurnResult()
    lm = IncrementalLM(list(prompt_ids))
    turn_tokens = 0
    M = st.M
    crossed_g2 = False          # belief ever passed g2 (decorative marker)
    crossed_g1 = False          # belief ever passed g1 (decorative marker)
    try:
        # symbol to emit: while not "confirming", posterior-match the payload;
        # once belief crosses g1 we switch to emitting the antipodal ACK symbol
        # for the current argmax candidate (confirmation), but this never ends
        # the turn — it only keeps conf_rho evolving.
        while turn_tokens < max_turn_tokens and st.tokens_used < MAX_PAYLOAD_TOKENS:
            probs = lm.probs()
            # Stop the turn when the BASE model genuinely wants to end it, i.e.
            # EOS is the most likely next token. (The old code sampled a
            # throwaway token and only stopped if that random draw was EOS,
            # which let steering run past the natural ending into the next
            # turn's role header — producing 'assistant' seams and run-on
            # multi-reply bubbles.) Decode happens at this EOS per the model.
            if int(torch.argmax(probs).item()) in CTX.eos_ids:
                res.ended_by_eos = True
                break

            # Mask chat-template boundary/role tokens so they can never be
            # emitted as watermark carriers (defense in depth against seams).
            if CTX.boundary_ids:
                bidx = torch.tensor(sorted(CTX.boundary_ids),
                                    device=probs.device, dtype=torch.long)
                probs = probs.clone()
                probs[bidx] = 0.0
                s = probs.sum()
                probs = probs / s if s > 0 else probs

            res.entropies.append(dist_entropy_bits(probs))
            key_ctx = _context_tokens_for_step(history_tokens,
                                               ARC_CONFIG.context_width)

            if st.in_confirmation:
                # confirmation: emit antipodal ACK/NACK symbol at p=2
                tx_symbol = SYM_ACK if st.conf_true_bit == 0 else SYM_NACK
                x, blp = emit_token(probs, key_ctx, tx_symbol,
                                    alphabet_size=P_CONF)
                angle = read_symbol_angle(x, key_ctx)
                history_tokens.append(x)
                res.tokens.append(x); res.base_logprobs.append(blp)
                lm.advance(x); turn_tokens += 1; st.tokens_used += 1
                ell = conf_symbol_likelihood(angle)
                st.conf_rho = st.conf_rho * ell
                st.conf_rho /= st.conf_rho.sum()
                # confirmation no longer terminates; if NACK wins, drop the
                # candidate's mass and return to message-matching mode.
                if st.conf_rho[1] >= rn:
                    st.knockdown[st.conf_cand] *= (1 - rn) / rn
                    st.pi = st.pi * st.knockdown; st.pi /= st.pi.sum()
                    st.knockdown = np.ones(st.M)
                    st.in_confirmation = False
                continue

            # R_t: synchronized, transcript-derived posterior-matching
            # randomness Rand(Lambda_t^{(3)}), NOT a local np.random draw.
            _, _, R = side_info_for_step(key_ctx)
            u = posterior_match_symbol(st.pi, st.payload, R)
            x, blp = emit_token(probs, key_ctx, u)
            angle = read_symbol_angle(x, key_ctx)
            history_tokens.append(x)
            res.tokens.append(x); res.base_logprobs.append(blp)
            lm.advance(x); turn_tokens += 1; st.tokens_used += 1

            ells = per_symbol_likelihood(angle)
            q = message_likelihood(ells, st.pi)
            st.pi = st.pi * q
            s = st.pi.sum()
            st.pi = st.pi / s if s > 0 else np.ones(M) / M

            eff = st.pi * st.knockdown; eff = eff / eff.sum()
            if eff.max() >= g2:
                crossed_g2 = True          # decorative: do NOT end the turn
            if eff.max() >= g1 and not st.in_confirmation:
                crossed_g1 = True
                cand = int(eff.argmax())
                st.in_confirmation = True
                st.conf_cand = cand
                st.conf_true_bit = 0 if cand == st.payload else 1
                st.conf_rho = np.array([0.5, 0.5])

        # --- EOS (or cap): decide between decode, or confidence-gated spill ---
        eff = st.pi * st.knockdown
        eff = eff / eff.sum() if eff.sum() > 0 else np.ones(M) / M
        # relevant belief: ACK-belief while confirming, else top-message belief
        if st.in_confirmation:
            belief = float(st.conf_rho[0])     # rho_ACK
        else:
            belief = float(eff.max())
        hit_cap = (st.tokens_used >= MAX_PAYLOAD_TOKENS) and not res.ended_by_eos

        # Spill if belief is weak AND we still have spill budget AND we stopped
        # at a real EOS (a token-cap stop always force-decodes — no runway left).
        if (belief < SPILL_BELIEF_THR and st.n_spills < MAX_SPILLS
                and res.ended_by_eos and not hit_cap):
            st.n_spills += 1
            res.completed = False              # payload NOT resolved this turn
            st.in_progress = True              # stays pending -> partner fillers
            res.outcome = "spill_lowconf"
            res.n_tokens = turn_tokens
            res.n_pad_tokens = 0
            return res

        # Otherwise decode argmax (belief ok, or spill budget exhausted, or cap).
        res.completed = True
        res.decoder_decoded = int(eff.argmax())
        res.mis_ack = (res.decoder_decoded != st.payload)
        st.in_progress = False
        if not res.ended_by_eos:
            res.outcome = "cap"                # hit token cap before EOS
        elif st.n_spills >= MAX_SPILLS and belief < SPILL_BELIEF_THR:
            res.outcome = "forced_maxspill"    # spill budget exhausted, low conf
        else:
            res.outcome = "eos_g2" if crossed_g2 else ("eos_g1" if crossed_g1
                                                       else "eos")
        res.n_tokens = turn_tokens
        res.n_pad_tokens = 0       # no separate padding phase in EOS-decode model
        return res
    finally:
        lm.free()


@torch.no_grad()
def _pad_to_eos(res, history_tokens, lm, max_turn_tokens):
    pad = 0
    total_emitted = res.n_tokens
    while total_emitted + pad < max_turn_tokens:
        probs = lm.probs()
        tok = int(torch.multinomial(probs, num_samples=1).item())
        if tok in CTX.eos_ids:
            break
        history_tokens.append(tok)
        res.tokens.append(tok)
        lm.advance(tok)
        pad += 1
    res.n_pad_tokens = pad


def _resume_confirmation(st, history_tokens, res, lm, max_turn_tokens,
                         g1, g2, ra, rn, turn_tokens_start=None):
    tx_symbol = SYM_ACK if st.conf_true_bit == 0 else SYM_NACK
    turn_tokens = res.n_tokens if turn_tokens_start is None else turn_tokens_start
    steps = 0
    while steps < MAX_CONF_STEPS and st.tokens_used < MAX_PAYLOAD_TOKENS \
            and turn_tokens < max_turn_tokens:
        probs = lm.probs()
        base_tok = int(torch.multinomial(probs, num_samples=1).item())
        if base_tok in CTX.eos_ids:
            res.ended_by_eos = True
            res.outcome = "eos"
            res.n_tokens = turn_tokens
            return res

        key_ctx = _context_tokens_for_step(history_tokens,
                                           ARC_CONFIG.context_width)
        res.entropies.append(dist_entropy_bits(probs))
        x, blp = emit_token(probs, key_ctx, tx_symbol, alphabet_size=P_CONF)
        angle = read_symbol_angle(x, key_ctx)
        history_tokens.append(x)
        res.tokens.append(x)
        res.base_logprobs.append(blp)
        lm.advance(x)
        turn_tokens += 1
        st.tokens_used += 1
        steps += 1

        ell = conf_symbol_likelihood(angle)
        st.conf_rho = st.conf_rho * ell
        st.conf_rho /= st.conf_rho.sum()

        if st.conf_rho[0] >= ra:
            res.completed = True
            res.outcome = "ack"
            res.decoder_decoded = st.conf_cand
            res.mis_ack = (st.conf_cand != st.payload)
            st.in_progress = False
            st.in_confirmation = False
            res.n_tokens = turn_tokens
            _pad_to_eos(res, history_tokens, lm, max_turn_tokens)
            return res
        if st.conf_rho[1] >= rn:
            st.knockdown[st.conf_cand] *= (1 - rn) / rn
            st.pi = st.pi * st.knockdown; st.pi /= st.pi.sum()
            st.knockdown = np.ones(st.M)
            st.in_confirmation = False
            res.n_tokens = turn_tokens
            return None

    res.n_tokens = turn_tokens
    if turn_tokens >= max_turn_tokens:
        return res
    return None


# ============================================================================
# Unwatermarked filler turn  (cap from profile). Returns emitted tokens; the
# dialogue driver counts these as WASTED tokens (they carry no payload).
# ============================================================================
@torch.no_grad()
def run_filler_turn(prompt_ids, history_tokens, max_filler_tokens):
    lm = IncrementalLM(list(prompt_ids))
    emitted = []
    try:
        for _ in range(max_filler_tokens):
            probs = lm.probs()
            tok = int(torch.multinomial(probs, num_samples=1).item())
            if tok in CTX.eos_ids:
                break
            emitted.append(tok)
            history_tokens.append(tok)
            lm.advance(tok)
        return emitted
    finally:
        lm.free()


@torch.no_grad()
def baseline_logprobs(prompt_ids, n_tokens):
    if n_tokens <= 0:
        return []
    lm = IncrementalLM(list(prompt_ids))
    logps = []
    try:
        for _ in range(n_tokens):
            probs = lm.probs()
            topv, topi = torch.topk(probs, TOP_K)
            trunc = torch.zeros_like(probs)
            trunc[topi] = topv
            trunc = trunc / trunc.sum()
            tok = int(torch.multinomial(trunc, num_samples=1).item())
            if tok in CTX.eos_ids:
                break
            logps.append(float(torch.log(probs[tok].clamp_min(1e-30)).item()))
            lm.advance(tok)
        return logps
    finally:
        lm.free()


# ============================================================================
# Dialogue driver — now parameterized by (task_gen, profile)
# ============================================================================
def decode_turn_text(token_ids):
    return CTX.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def run_dialogue(rng, task_gen: TaskGen, profile: ConvProfile, seed=None):
    if seed is None:
        seed = {"topic": getattr(profile, "topic", ""),
                "opener": getattr(profile, "opener", "")}
    topic = seed["topic"]
    opener = seed["opener"]
    task_gen.reset(rng)
    history = [{"agent": 0, "content": opener, "kind": "opener"}]
    history_tokens = CTX.tokenizer.encode(opener, add_special_tokens=False)

    pending = [None, None]
    completed_payloads = 0
    turn_idx = 0
    correct = 0
    errors = 0
    misacks = 0
    unfinished = 0
    wm_tokens_total = 0
    pad_tokens_total = 0
    filler_tokens_total = 0       # tokens emitted on filler turns (wasted)
    coded_bits_correct = 0.0     # sum of coded bits over CORRECTLY decoded payloads
    coded_bits_attempted = 0.0   # sum over all completed payloads
    wm_logprob_sum = 0.0
    wm_token_count = 0
    entropy_sum = 0.0             # sum of per-step unwm entropy (bits)
    entropy_count = 0            # number of watermarked emission steps
    entropy_sum_early = 0.0      # sum over first 20 tokens of each turn
    entropy_count_early = 0      # count over first 20 tokens of each turn
    wm_turn_lengths = []         # carrier-token count per watermarked turn
    bl_logprob_sum = 0.0
    bl_token_count = 0
    filler_turns = 0
    resumes = 0

    first_speaker = 1
    max_turn = profile.max_turn_tokens
    max_fill = profile.max_filler_tokens

    while turn_idx < ROUNDS_PER_DIALOGUE:
        agent = (first_speaker + turn_idx) % 2
        other = 1 - agent

        if pending[other] is not None and pending[other].in_progress:
            prompt_ids = CTX.render_dialogue(history, agent, profile, topic)
            ftoks = run_filler_turn(prompt_ids, history_tokens, max_fill)
            history.append({"agent": agent,
                            "content": decode_turn_text(ftoks) or "(…)",
                            "kind": "filler",
                            "wm_tokens": 0,
                            "pad_tokens": 0,
                            "filler_tokens": int(len(ftoks))})
            filler_turns += 1
            filler_tokens_total += len(ftoks)
            turn_idx += 1
            continue

        if pending[agent] is not None and pending[agent].in_progress:
            st = pending[agent]
            resumes += 1
            turn_was_resume = True
        else:
            payload, M_msg, coded_bits, move_label = task_gen.next_payload(rng)
            st = PayloadState(payload, M_msg, coded_bits, move_label)
            pending[agent] = st
            turn_was_resume = False

        prompt_ids = CTX.render_dialogue(history, agent, profile, topic)
        res = run_watermarked_turn(prompt_ids, history_tokens, st, max_turn,
                                   BAM_G1, BAM_G2, BAM_RACK, BAM_RNACK)
        turn_idx += 1
        wm_tokens_total += res.n_tokens
        pad_tokens_total += res.n_pad_tokens
        wm_logprob_sum += sum(res.base_logprobs)
        wm_token_count += len(res.base_logprobs)
        entropy_sum += sum(res.entropies)
        entropy_count += len(res.entropies)
        entropy_sum_early += sum(res.entropies[:20])
        entropy_count_early += len(res.entropies[:20])
        if res.n_tokens > 0:
            wm_turn_lengths.append(res.n_tokens)

        if len(res.base_logprobs) > 0:
            bl = baseline_logprobs(prompt_ids, len(res.base_logprobs))
            bl_logprob_sum += sum(bl)
            bl_token_count += len(bl)

        history.append({
            "agent": agent,
            "content": decode_turn_text(res.tokens) or "(…)",
            "kind": "watermarked",
            "resumed": bool(turn_was_resume),
            "payload": int(st.payload),
            "M": int(st.M),
            "coded_bits": float(st.coded_bits),
            "move_label": st.label,
            "completed": bool(res.completed),
            "decoded": int(res.decoder_decoded),
            "correct": bool(res.completed and res.decoder_decoded == st.payload),
            "outcome": res.outcome,
            "wm_tokens": int(res.n_tokens),
            "pad_tokens": int(res.n_pad_tokens),
            "payload_tokens_so_far": int(st.tokens_used),
        })

        if res.completed:
            pending[agent] = None
            completed_payloads += 1
            coded_bits_attempted += st.coded_bits
            decoded = res.decoder_decoded
            if decoded == st.payload:
                correct += 1
                coded_bits_correct += st.coded_bits
            else:
                errors += 1
            if res.mis_ack:
                misacks += 1

    for st in pending:
        if st is not None and st.in_progress:
            unfinished += 1

    perplexity = (math.exp(-wm_logprob_sum / wm_token_count)
                  if wm_token_count > 0 else float("nan"))
    perplexity_baseline = (math.exp(-bl_logprob_sum / bl_token_count)
                           if bl_token_count > 0 else float("nan"))
    total_tok = wm_tokens_total + pad_tokens_total
    bits_per_token = coded_bits_correct / max(total_tok, 1)
    # pad_ratio now counts filler-turn tokens as wasted capacity alongside pad.
    wasted_tokens = pad_tokens_total + filler_tokens_total
    pad_ratio = wasted_tokens / max(wm_tokens_total, 1)
    avg_entropy = (entropy_sum / entropy_count) if entropy_count > 0 else float("nan")
    avg_entropy_early = (entropy_sum_early / entropy_count_early
                         if entropy_count_early > 0 else float("nan"))
    avg_wm_turn_len = (float(np.mean(wm_turn_lengths))
                       if wm_turn_lengths else float("nan"))

    return {
        "rounds": turn_idx,
        "completed_payloads": completed_payloads,
        "unfinished": unfinished,
        "correct": correct,
        "errors": errors,
        "misacks": misacks,
        "filler_turns": filler_turns,
        "resumes": resumes,
        "wm_tokens_total": wm_tokens_total,
        "pad_tokens_total": pad_tokens_total,
        "filler_tokens_total": filler_tokens_total,
        "coded_bits_correct": coded_bits_correct,
        "coded_bits_attempted": coded_bits_attempted,
        "bits_per_token": bits_per_token,
        "pad_ratio": pad_ratio,
        "spill": resumes + unfinished,
        "avg_wm_tokens_per_payload": wm_tokens_total / max(correct + errors, 1),
        "avg_wm_turn_len": avg_wm_turn_len,
        "avg_entropy_bits": avg_entropy,
        "entropy_count": entropy_count,
        "avg_entropy_bits_early": avg_entropy_early,
        "entropy_count_early": entropy_count_early,
        "perplexity": perplexity,
        "perplexity_baseline": perplexity_baseline,
        "seed_topic": topic,
        "seed_opener": opener,
        "history": history,
    }


# ============================================================================
# Per-cell run (one task x one conversation)
# ============================================================================
def run_cell(task_name, conv_name):
    task_gen = TASKS[task_name]()
    profile = CONVERSATIONS[conv_name]
    per_dialogue = []
    log(f"\n  --- CELL  task={task_name:<9} conv={conv_name:<13} "
        f"(max_turn={profile.max_turn_tokens}) ---")
    for di in range(N_DIALOGUES):
        rng = np.random.RandomState(70000 + di)
        np.random.seed(70000 + di)
        torch.manual_seed(70000 + di)
        t0 = time.time()
        seed = pick_seed(profile, di)
        m = run_dialogue(rng, task_gen, profile, seed)
        dt = time.time() - t0
        per_dialogue.append(m)
        log(f"    d{di+1:>2}: payld={m['completed_payloads']:>2} ok={m['correct']:>2} "
            f"err={m['errors']:>2} "
            f"{'[clean]' if m['errors']==0 else '[ERR]'} "
            f"bits/tok={m['bits_per_token']:.4f} "
            f"tok/payld={m['avg_wm_tokens_per_payload']:.0f} "
            f"[{dt:.1f}s]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg_wm   = sum(d["wm_tokens_total"] for d in per_dialogue)
    agg_pad  = sum(d["pad_tokens_total"] for d in per_dialogue)
    agg_fill_tok = sum(d["filler_tokens_total"] for d in per_dialogue)
    agg_cbc  = sum(d["coded_bits_correct"] for d in per_dialogue)
    agg_cor  = sum(d["correct"] for d in per_dialogue)
    agg_err  = sum(d["errors"] for d in per_dialogue)
    agg_unf  = sum(d["unfinished"] for d in per_dialogue)
    agg_res  = sum(d["resumes"] for d in per_dialogue)
    agg_fill = sum(d["filler_turns"] for d in per_dialogue)
    bits_per_token = agg_cbc / max(agg_wm + agg_pad, 1)
    err_rate = agg_err / max(agg_cor + agg_err, 1)
    # completion rate = fraction of DIALOGUES that are fully clean: every
    # decoded message correct AND no payload left unfinished (a payload can now
    # spill on the final round and never resolve). Distinct from err_rate, which
    # is message-level over completed payloads only.
    n_dialogues = len(per_dialogue)
    # dialogue-level error rate = fraction of dialogues containing >=1 wrong
    # decoded message (i.e. 1 - completion rate). Unfinished payloads do not
    # count against it; only actual decode errors do.
    dlg_with_error = sum(1 for d in per_dialogue if d["errors"] > 0)
    dialogue_err_rate = dlg_with_error / max(n_dialogues, 1)
    # efficiency = fraction of ALL emitted tokens that carry payload (carriers).
    # Under pure force-decode (no spill) every token is a carrier -> 1.0; with
    # spill, filler turns carry nothing, so efficiency drops below 1.0.
    payload_tokens = agg_wm                       # watermarked carrier tokens
    all_emitted = agg_wm + agg_pad + agg_fill_tok  # carriers + pad + filler
    efficiency = payload_tokens / max(all_emitted, 1)
    avg_payloads = float(np.mean([d["completed_payloads"] for d in per_dialogue]))

    # ---- Standard errors (std / sqrt(N)) across the N dialogues in this cell ----
    # Each metric is computed per-dialogue, then SE = sample_std / sqrt(N).
    N = max(n_dialogues, 1)
    def _se(values):
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return 0.0
        return float(np.std(v, ddof=1) / math.sqrt(len(v)))
    # payloads/dialogue: per-dialogue completed count
    se_avg_payloads = _se([d["completed_payloads"] for d in per_dialogue])
    # error rate: per-dialogue message-level error fraction
    per_dlg_err = [d["errors"] / max(d["correct"] + d["errors"], 1)
                   for d in per_dialogue]
    se_err_rate = _se(per_dlg_err)
    # dialogue error rate: per-dialogue 0/1 indicator of "has >=1 error"
    per_dlg_has_error = [1.0 if d["errors"] > 0 else 0.0 for d in per_dialogue]
    se_dialogue_err_rate = _se(per_dlg_has_error)
    # efficiency: per-dialogue carrier / all-emitted fraction
    per_dlg_eff = [d["wm_tokens_total"] /
                   max(d["wm_tokens_total"] + d["pad_tokens_total"]
                       + d["filler_tokens_total"], 1)
                   for d in per_dialogue]
    se_efficiency = _se(per_dlg_eff)
    # bits/token: per-dialogue effective payload bits per emitted token
    se_bits_per_token = _se([d["bits_per_token"] for d in per_dialogue])
    # token-weighted mean entropy across all watermarked emission steps in the cell
    ent_count = sum(d["entropy_count"] for d in per_dialogue)
    ent_weighted = sum(d["avg_entropy_bits"] * d["entropy_count"]
                       for d in per_dialogue
                       if not math.isnan(d["avg_entropy_bits"]))
    avg_entropy = (ent_weighted / ent_count) if ent_count > 0 else float("nan")
    # same, restricted to first 20 tokens of each turn (early-token entropy)
    ent_count_e = sum(d["entropy_count_early"] for d in per_dialogue)
    ent_weighted_e = sum(d["avg_entropy_bits_early"] * d["entropy_count_early"]
                         for d in per_dialogue
                         if not math.isnan(d["avg_entropy_bits_early"]))
    avg_entropy_early = (ent_weighted_e / ent_count_e) if ent_count_e > 0 else float("nan")
    # mean watermarked-turn length, weighted by number of such turns
    tl_vals = [d["avg_wm_turn_len"] for d in per_dialogue
               if not math.isnan(d["avg_wm_turn_len"])]
    avg_wm_turn_len = float(np.mean(tl_vals)) if tl_vals else float("nan")
    ppls = [d["perplexity"] for d in per_dialogue if not math.isnan(d["perplexity"])]
    ppls_b = [d["perplexity_baseline"] for d in per_dialogue
              if not math.isnan(d["perplexity_baseline"])]
    return {
        "task": task_name, "conv": conv_name,
        "max_turn_tokens": profile.max_turn_tokens,
        "avg_payloads_per_dialogue": avg_payloads,
        "bits_per_token": bits_per_token,
        "dialogue_err_rate": dialogue_err_rate,
        "dialogues_with_error": dlg_with_error,
        "n_dialogues": n_dialogues,
        "efficiency": efficiency,
        "se_avg_payloads": se_avg_payloads,
        "se_err_rate": se_err_rate,
        "se_dialogue_err_rate": se_dialogue_err_rate,
        "se_efficiency": se_efficiency,
        "se_bits_per_token": se_bits_per_token,
        "unfinished": agg_unf, "resumes": agg_res, "filler_turns": agg_fill,
        "filler_tokens_total": agg_fill_tok,
        "correct": agg_cor, "errors": agg_err, "err_rate": err_rate,
        "wm_tokens_total": agg_wm, "pad_tokens_total": agg_pad,
        "coded_bits_correct": agg_cbc,
        "avg_entropy_bits": avg_entropy,
        "avg_wm_turn_len": avg_wm_turn_len,
        "entropy_count": ent_count,
        "avg_entropy_bits_early": avg_entropy_early,
        "entropy_count_early": ent_count_e,
        "perplexity": float(np.mean(ppls)) if ppls else float("nan"),
        "perplexity_baseline": float(np.mean(ppls_b)) if ppls_b else float("nan"),
        "per_dialogue": per_dialogue,
    }

# ============================================================================
# Run ALL models over the SAME dialogues (each: 3x3 task x conversation matrix)
# ============================================================================
# The per-dialogue seeding in run_cell (RandomState / np / torch all seeded by
# 70000 + di, and the scenario via pick_seed(profile, di)) depends ONLY on the
# dialogue index di, NOT on the model. Driving every model through the identical
# (task, conv, di) grid therefore reuses the SAME scenario, opener, and secret
# payload / game trajectory for all three models: the runs are paired
# dialogue-for-dialogue rather than independent. (The generated *text* still
# differs per model -- that is unavoidable -- but the experimental conditions
# each dialogue is run under are shared.) Results are stored per model instead of
# being overwritten, and each model gets its own tagged CSV / PNG / rollout
# files, plus a combined long-format CSV with a `model` column.
import json
import html as _html

TASK_ORDER = ["tictactoe", "chess", "go19"]          # low -> high coded bits
CONV_ORDER = ["chat", "debate"]                      # short -> long turns

# Which (conv, task) cells to feature in the rollout figure.
FIG_CELLS = [
    ("chat",   "tictactoe"),
    ("debate", "chess"),
]
FIG_N_ROLLOUTS = 3          # dialogues to show per cell


def _payload_label(task_name, turn):
    """Human-readable transmitted message for a completed-payload turn.
    Prefers the real move string captured at generation time (e.g. 'e2e4',
    '(2,3)', 'Q16'); falls back to an index form if unavailable."""
    lbl = turn.get("move_label", "")
    if lbl:
        return lbl
    idx = turn["payload"]
    n = turn["M"] - 1
    return f"msg {idx}/{n}"


def _select_correct_dialogues(cell, k):
    """Return up to k per-dialogue dicts whose every completed payload decoded
    correctly (errors == 0 and at least one completed payload)."""
    out = []
    for d in cell["per_dialogue"]:
        if d["errors"] == 0 and d["correct"] >= 1:
            out.append(d)
        if len(out) >= k:
            break
    return out


def build_rollout_records():
    records = []
    for conv_name, task_name in FIG_CELLS:
        cell = all_cells.get((task_name, conv_name))
        if cell is None:
            continue
        chosen = _select_correct_dialogues(cell, FIG_N_ROLLOUTS)
        cell_rollouts = []
        for d in chosen:
            turns = []
            for turn in d["history"]:
                kind = turn.get("kind", "watermarked")
                rec = {
                    "agent": turn["agent"],
                    "content": turn["content"],
                    "kind": kind,
                }
                if kind == "watermarked":
                    completed = turn.get("completed", False)
                    wm = max(turn.get("wm_tokens", 0), 0)
                    pad = max(turn.get("pad_tokens", 0), 0)
                    total = wm + pad
                    rec.update({
                        "completed": completed,
                        "resumed": turn.get("resumed", False),
                        "outcome": turn.get("outcome", ""),
                        "wm_tokens": wm,
                        "pad_tokens": pad,
                        "coded_bits": turn.get("coded_bits", 0.0),
                        "correct": turn.get("correct", False),
                        "message": _payload_label(task_name, turn),
                        "payload_tokens_so_far": turn.get("payload_tokens_so_far", 0),
                    })
                    if completed:
                        # carrier tokens / total emitted tokens this turn
                        rec["tok_ratio"] = (wm / total) if total > 0 else 0.0
                        rec["tok_use"] = f"{wm}/{total}"
                    else:
                        # spilled / in-progress: no pad phase, all tokens are
                        # carriers; payload continues on a later turn.
                        rec["tok_ratio"] = (wm / total) if total > 0 else 0.0
                        rec["tok_use"] = f"{wm}/{total}"
                elif kind == "filler":
                    fl = int(turn.get("filler_tokens", 0))
                    rec.update({
                        "filler_tokens": fl,
                        "tok_use": f"0/{fl}",
                        "tok_ratio": 0.0,
                    })
                turns.append(rec)
            cell_rollouts.append({
                "correct": d["correct"],
                "completed_payloads": d["completed_payloads"],
                "errors": d["errors"],
                "turns": turns,
            })
        records.append({
            "conv": conv_name,
            "task": task_name,
            "max_turn_tokens": cell["max_turn_tokens"],
            "rollouts": cell_rollouts,
        })
    return records


def render_rollouts_html(records):
    def esc(s):
        return _html.escape(str(s)).replace("\n", "<br>")

    css = """
    <style>
      body { background:#0b0b0c; color:#e8e8e8; font-family:-apple-system,
             Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:24px; }
      h1 { font-size:20px; font-weight:600; }
      h2 { font-size:16px; font-weight:600; margin:28px 0 8px; color:#bdbdbd; }
      .cell { margin-bottom:40px; }
      .rollouts { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; }
      .panel { background:#161618; border:1px solid #2a2a2e; border-radius:12px;
               padding:14px 16px 18px; width:420px; box-sizing:border-box; }
      .panel-head { display:flex; justify-content:space-between; font-size:12px;
                    color:#9a9a9a; margin-bottom:12px; }
      .names { display:flex; justify-content:space-between; font-size:12px;
               color:#cfcfcf; font-weight:600; padding:0 4px 8px;
               border-bottom:1px solid #2a2a2e; margin-bottom:10px; }
      /* one full-width column; bubbles align L (A) or R (B) within it */
      .row { display:flex; margin:8px 0; }
      .row.left  { justify-content:flex-start; }
      .row.right { justify-content:flex-end; }
      .wrap { max-width:74%; display:flex; flex-direction:column; }
      .row.left  .wrap { align-items:flex-start; }
      .row.right .wrap { align-items:flex-end; }
      .bub { padding:8px 12px; border-radius:16px; font-size:12.5px;
             line-height:1.38; word-wrap:break-word; overflow-wrap:anywhere; }
      .a { background:#2f7d4f; color:#fff; border-bottom-left-radius:4px; }
      .b { background:#26262b; color:#ededed; border-bottom-right-radius:4px; }
      .ann { font-size:10.5px; margin-top:4px;
             font-variant-numeric:tabular-nums; }
      .ann .msg { color:#ffd479; }
      .ann .tok { color:#86c5ff; margin-left:8px; }
      .rule { margin-left:8px; padding:1px 6px; border-radius:6px;
              font-size:9.5px; font-weight:600; }
      .rule-c1 { background:#1f3a2a; color:#7fdca0; }   /* early-stop→EOS */
      .rule-c2 { background:#3a2a1f; color:#f0b27a; }   /* force-decode  */
      .ann-spill .msg { color:#e0a3ff; }     /* purple = payload spilling */
      .ann-spill .tok { color:#9a7bbf; }
      .ann-filler .msg { color:#777; }       /* grey = filler, no payload */
      .ann-filler .tok { color:#666; }
      .filler .bub { opacity:0.45; font-style:italic; }
      .legend { font-size:11px; color:#888; margin-top:6px; max-width:900px; }
    </style>
    """

    parts = ["<html><head><meta charset='utf-8'>", css, "</head><body>"]
    parts.append("<h1>Turn-based covert channel — example rollouts</h1>")
    parts.append("<div class='legend'>Green (left) = Agent A, dark (right) = "
                 "Agent B, in conversation order. "
                 "<span style='color:#ffd479'>▣ message ✓/✗ + ⛁ ratio</span> = "
                 "payload decoded on this turn (carrier/total tokens). Each "
                 "decoded message is tagged with which EOS rule produced it: "
                 "<span style='background:#1f3a2a;color:#7fdca0;padding:1px 6px;"
                 "border-radius:6px;font-size:10px'>C1 · early-stop→EOS</span> "
                 "= belief converged before the natural EOS, so generation "
                 "continued to EOS and then decoded (Change 1); "
                 "<span style='background:#3a2a1f;color:#f0b27a;padding:1px 6px;"
                 "border-radius:6px;font-size:10px'>C2 · force-decode</span> "
                 "= EOS arrived before belief converged, so the highest-belief "
                 "message was force-decoded at EOS (Change 2). "
                 "<span style='color:#777'>— filler</span> = unwatermarked turn "
                 "(no payload).</div>")

    for rec in records:
        parts.append("<div class='cell'>")
        parts.append(f"<h2>{esc(rec['conv'])} &times; {esc(rec['task'])} "
                     f"(max {rec['max_turn_tokens']} tok/turn)</h2>")
        parts.append("<div class='rollouts'>")
        for ri, roll in enumerate(rec["rollouts"], 1):
            parts.append("<div class='panel'>")
            parts.append(
                f"<div class='panel-head'><span>rollout {ri}</span>"
                f"<span>{roll['correct']} msgs ok &nbsp; "
                f"{'✓ clean' if roll.get('errors', 0) == 0 else '✗ has error'}"
                f"</span></div>")
            parts.append("<div class='names'><span>Agent A</span>"
                         "<span>Agent B</span></div>")
            # Single full-width stream in conversation order. Agent A bubbles
            # align left, Agent B bubbles align right (chat-app style).
            for turn in roll["turns"]:
                is_a = (turn["agent"] == 0)
                side = "left" if is_a else "right"
                bub_cls = "a" if is_a else "b"
                kind = turn.get("kind", "")
                filler = " filler" if kind == "filler" else ""
                ann = ""
                if kind == "watermarked":
                    msg = esc(turn.get("message", ""))
                    tok = esc(turn.get("tok_use", ""))
                    pct = turn.get("tok_ratio", 0) * 100
                    if turn.get("completed"):
                        ok = "✓" if turn.get("correct") else "✗"
                        resumed = "↪ " if turn.get("resumed") else ""
                        # Which rule fired:
                        #   converged before EOS (eos_g1/eos_g2) -> Change 1
                        #     (early-stop removed; ran to EOS then decoded)
                        #   not converged at EOS (eos/cap)        -> Change 2
                        #     (forced decode at EOS)
                        oc = turn.get("outcome", "")
                        if oc in ("eos_g1", "eos_g2"):
                            rule = ("<span class='rule rule-c1'>"
                                    "C1 · early-stop→EOS</span>")
                        else:
                            rule = ("<span class='rule rule-c2'>"
                                    "C2 · force-decode</span>")
                        ann = (f"<div class='ann'>"
                               f"<span class='msg'>{resumed}▣ {msg} {ok}</span>"
                               f"<span class='tok'>⛁ {tok} ({pct:.0f}%)</span>"
                               f"{rule}</div>")
                    else:
                        # spilled: payload still transmitting, continues later
                        sofar = turn.get("payload_tokens_so_far", 0)
                        ann = (f"<div class='ann ann-spill'>"
                               f"<span class='msg'>↻ {msg} carrying…</span>"
                               f"<span class='tok'>⛁ {tok} ({pct:.0f}%) · "
                               f"{sofar} tok so far</span></div>")
                elif kind == "filler":
                    fl = esc(turn.get("tok_use", ""))
                    ann = (f"<div class='ann ann-filler'>"
                           f"<span class='msg'>— filler (no payload)</span>"
                           f"<span class='tok'>⛁ {fl}</span></div>")
                parts.append(
                    f"<div class='row {side}{filler}'><div class='wrap'>"
                    f"<div class='bub {bub_cls}'>{esc(turn['content'])}</div>"
                    f"{ann}</div></div>")
            parts.append("</div>")  # panel
        parts.append("</div>")  # rollouts
        parts.append("</div>")  # cell
    parts.append("</body></html>")
    return "".join(parts)

def write_rollouts(tag):
    """Write the rollout figure JSON/HTML for the current model's `all_cells`."""
    out_json = "rollouts_%s.json" % tag
    out_html = "rollouts_%s.html" % tag
    try:
        _fig_records = build_rollout_records()
        with open(out_json, "w") as f:
            json.dump(_fig_records, f, indent=2, ensure_ascii=False)
        with open(out_html, "w") as f:
            f.write(render_rollouts_html(_fig_records))
        n_found = {f"{r['conv']}-{r['task']}": len(r["rollouts"]) for r in _fig_records}
        log(f"\nWrote {out_json} and {out_html}")
        log(f"  rollouts found per cell (want {FIG_N_ROLLOUTS}): {n_found}")
        for r in _fig_records:
            if len(r["rollouts"]) < FIG_N_ROLLOUTS:
                log(f"  WARNING: only {len(r['rollouts'])} all-correct dialogues for "
                    f"{r['conv']}-{r['task']} (need {FIG_N_ROLLOUTS}); "
                    f"increase N_DIALOGUES or relax the all-correct filter.")
    except Exception as _e:
        log(f"\n[rollout figure] skipped due to error: {_e}")


def print_matrix(title, key, fmt="{:>8.4f}", se_key=None):
    log(f"\n{title}")
    header_label = "task\\conv"
    log(f"{header_label:<14}" + "".join(f"{c:>18}" for c in CONV_ORDER))
    for t in TASK_ORDER:
        row = f"{t:<14}"
        for c in CONV_ORDER:
            cell = all_cells[(t, c)]
            if se_key is not None:
                val = fmt.format(cell[key]).strip()
                se = fmt.format(cell[se_key]).strip()
                row += f"{val}±{se}".rjust(18)
            else:
                row += fmt.format(cell[key]).rjust(18)
        log(row)

# ---- Shared CSV schema (identical fields to the single-model script) ----
_CSV_HEADER = (
    "task,conv,max_turn_tokens,avg_payloads_per_dialogue,se_avg_payloads,"
    "bits_per_token,se_bits_per_token,"
    "dialogue_err_rate,se_dialogue_err_rate,efficiency,se_efficiency,"
    "dialogues_with_error,n_dialogues,"
    "filler_turns,filler_tokens_total,"
    "correct,errors,err_rate,se_err_rate,wm_tokens_total,pad_tokens_total,"
    "coded_bits_correct,avg_wm_turn_len,avg_entropy_bits,"
    "perplexity_wm,perplexity_baseline\n"
)


def _csv_row(s):
    return (f"{s['task']},{s['conv']},{s['max_turn_tokens']},"
            f"{s['avg_payloads_per_dialogue']:.4f},{s['se_avg_payloads']:.4f},"
            f"{s['bits_per_token']:.6f},{s['se_bits_per_token']:.6f},"
            f"{s['dialogue_err_rate']:.4f},{s['se_dialogue_err_rate']:.4f},"
            f"{s['efficiency']:.4f},{s['se_efficiency']:.4f},"
            f"{s['dialogues_with_error']},"
            f"{s['n_dialogues']},"
            f"{s['filler_turns']},{s['filler_tokens_total']},"
            f"{s['correct']},{s['errors']},{s['err_rate']:.6f},"
            f"{s['se_err_rate']:.6f},"
            f"{s['wm_tokens_total']},{s['pad_tokens_total']},"
            f"{s['coded_bits_correct']:.4f},"
            f"{s['avg_wm_turn_len']:.4f},{s['avg_entropy_bits']:.4f},"
            f"{s['perplexity']:.4f},{s['perplexity_baseline']:.4f}\n")


def write_combined_csv(all_results, path="turnbased_bam_matrix_all.csv"):
    """One long-format CSV across all models (leading `model` column)."""
    with open(path, "w") as f:
        f.write("model," + _CSV_HEADER)
        for model_name in MODEL_NAMES:
            cells = all_results.get(model_name, {})
            tag = MODEL_TAGS.get(model_name, _slug(model_name))
            for t in TASK_ORDER:
                for c in CONV_ORDER:
                    s = cells.get((t, c))
                    if s is not None:
                        f.write(f"{tag}," + _csv_row(s))
    log(f"\nWrote {path}")


def _slug(model_name):
    return (model_name.split("/")[-1]
            .replace(".", "").replace("-", "").lower())


def report_model(tag, model_name):
    """Console matrices + per-model CSV + per-model heatmap for `all_cells`."""
    log("\n" + "=" * 80)
    log(f"3x3 MATRIX RESULTS  [{model_name}]  (rows: coded bits low->high; "
        f"cols: turn length short->long)")
    log("  values shown as mean+/-SE  (SE = std across dialogues / sqrt(N))")
    log("=" * 80)
    print_matrix("BITS PER TOKEN  (higher = better channel utilization)",
                 "bits_per_token", "{:>7.4f}", se_key="se_bits_per_token")
    print_matrix("ERR RATE  (message-level: fraction of decoded messages that are wrong)",
                 "err_rate", "{:>7.4f}", se_key="se_err_rate")
    print_matrix("DIALOGUE ERR RATE  (fraction of dialogues with >=1 wrong message; = 1 - completion rate)",
                 "dialogue_err_rate", "{:>6.4f}", se_key="se_dialogue_err_rate")
    print_matrix("EFFICIENCY  (fraction of emitted tokens that carry payload; 1.0 = no filler/pad waste)",
                 "efficiency", "{:>6.4f}", se_key="se_efficiency")
    print_matrix("AVG PAYLOADS / DIALOGUE",
                 "avg_payloads_per_dialogue", "{:>6.2f}", se_key="se_avg_payloads")

    # Per-conversation-setting summary: avg token length + avg unwm entropy.
    log("\n" + "-" * 80)
    log("PER-CONVERSATION SUMMARY  (avg watermarked-turn length; avg "
        "unwatermarked next-token entropy)")
    log("-" * 80)
    log(f"{'conv':<16}{'max_turn':>10}{'avg_tok_len':>14}"
        f"{'entropy_all':>16}{'entropy_early':>16}")
    for c in CONV_ORDER:
        cells = [all_cells[(t, c)] for t in TASK_ORDER]
        ent_n = sum(cl["entropy_count"] for cl in cells)
        ent_w = sum(cl["avg_entropy_bits"] * cl["entropy_count"]
                    for cl in cells if not math.isnan(cl["avg_entropy_bits"]))
        conv_entropy = (ent_w / ent_n) if ent_n > 0 else float("nan")
        ent_ne = sum(cl["entropy_count_early"] for cl in cells)
        ent_we = sum(cl["avg_entropy_bits_early"] * cl["entropy_count_early"]
                     for cl in cells if not math.isnan(cl["avg_entropy_bits_early"]))
        conv_entropy_early = (ent_we / ent_ne) if ent_ne > 0 else float("nan")
        tl = [cl["avg_wm_turn_len"] for cl in cells
              if not math.isnan(cl["avg_wm_turn_len"])]
        conv_tok_len = float(np.mean(tl)) if tl else float("nan")
        log(f"{c:<16}{CONVERSATIONS[c].max_turn_tokens:>10}"
            f"{conv_tok_len:>14.1f}{conv_entropy:>16.3f}{conv_entropy_early:>16.3f}")

    out_csv = "turnbased_bam_matrix_%s.csv" % tag
    with open(out_csv, "w") as f:
        f.write(_CSV_HEADER)
        for t in TASK_ORDER:
            for c in CONV_ORDER:
                f.write(_csv_row(all_cells[(t, c)]))
    log(f"\nWrote {out_csv}")

    # Heatmap of bits/token (tasks x conversations; grid is 3x2, not square).
    B = np.array([[all_cells[(t, c)]["bits_per_token"] for c in CONV_ORDER]
                  for t in TASK_ORDER])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(B, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CONV_ORDER)))
    ax.set_xticklabels([f"{c}\n(~{CONVERSATIONS[c].max_turn_tokens}tok)"
                        for c in CONV_ORDER])
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels([f"{t}\n(~{b}b)"
                        for t, b in zip(TASK_ORDER, ["2.3", "5", "8.4"])])
    ax.set_xlabel("conversation type (turn length ->)")
    ax.set_ylabel("embedding task (coded bits ->)")
    ax.set_title(f"Effective payload bits per emitted token\n{model_name}")
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            ax.text(j, i, f"{B[i, j]:.3f}", ha="center", va="center",
                    color="white" if B[i, j] < B.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="bits / token")
    out_plot = "turnbased_bam_matrix_%s.png" % tag
    plt.tight_layout(); plt.savefig(out_plot, dpi=140); plt.close(fig)
    log(f"Wrote {out_plot}")


# ---------------------------------------------------------------------------
# Main loop: one model resident at a time; identical dialogues across models.
# ---------------------------------------------------------------------------
ALL_RESULTS = {}
for model_name in MODEL_NAMES:
    tag = MODEL_TAGS.get(model_name, _slug(model_name))
    log("\n" + "#" * 72)
    log(f"# MODEL: {model_name}  ->  tag={tag}")
    log(f"#   tasks (coded bits): tictactoe~2.3  chess~5  go19~8.4")
    log(f"#   convs (tok/turn):   chat~90  debate~300")
    log(f"#   {N_DIALOGUES} dialogues x {ROUNDS_PER_DIALOGUE} rounds per cell")
    log(f"#   dialogues shared across models (seeded by index 70000+di only)")
    log("#" * 72)
    CTX = LMContext(model_name)          # module global; run_cell/run_dialogue use it
    all_cells = {}                       # module global; reset per model
    log(f"  (Laplace comm b={_B_LAPLACE:.4f} eps={EPS_NOISE}; "
        f"conf p={P_CONF} b_conf={_B_CONF:.4f} eps_conf={EPS_CONF})")
    try:
        for task_name in TASK_ORDER:
            for conv_name in CONV_ORDER:
                cell = run_cell(task_name, conv_name)
                all_cells[(task_name, conv_name)] = cell
                d0 = cell["per_dialogue"][0]
                log(f"    sample transcript (d1, first 4 turns):")
                for turn in d0["history"][:5]:
                    who = "A" if turn["agent"] == 0 else "B"
                    text = " ".join(turn["content"].split())[:160]
                    log(f"      [{who}] {text}")
        write_rollouts(tag)              # build_rollout_records reads all_cells
    finally:
        CTX.teardown()
        CTX = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report_model(tag, model_name)        # matrices + per-model CSV + heatmap
    ALL_RESULTS[model_name] = dict(all_cells)

# Combined long-format CSV across all models.
write_combined_csv(ALL_RESULTS)

# Compact cross-model comparison (bits/token and message err_rate per cell).
log("\n" + "=" * 80)
log("CROSS-MODEL COMPARISON  (paired dialogues; same scenarios/payloads per cell)")
log("=" * 80)
for t in TASK_ORDER:
    for c in CONV_ORDER:
        log(f"\n  {t} x {c}:")
        for model_name in MODEL_NAMES:
            s = ALL_RESULTS.get(model_name, {}).get((t, c))
            if s is None:
                continue
            tag = MODEL_TAGS.get(model_name, _slug(model_name))
            log(f"    {tag:<10} bits/tok={s['bits_per_token']:.4f}"
                f"+/-{s['se_bits_per_token']:.4f}   "
                f"msg_err={s['err_rate']:.4f}+/-{s['se_err_rate']:.4f}   "
                f"eff={s['efficiency']:.3f}   "
                f"avg_payload={s['avg_payloads_per_dialogue']:.2f}")
log("\nDone: all models complete.")