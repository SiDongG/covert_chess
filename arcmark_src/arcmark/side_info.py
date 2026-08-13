"""Pluggable side-information key generation for ArcMark.

This module centralises all side-information (s_index, perm_seed) generation
so that different strategies can be swapped in without touching the encoder
or decoder logic.

Supported modes (SideInfoMode):

1. ``hash_context``  (default / production)
   Keys derived by SHA-256 hashing the previous ``context_width`` raw
   token IDs together with the secret key.  Fast, but any token changed
   by an attack causes hash drift from that position onward.

2. ``normalized``
   Same as ``hash_context`` but the context tokens are decoded to text,
   lowercased, punctuation-stripped, and re-encoded before hashing.
   Robust to lowercase and misspelling attacks since surface changes
   that do not alter the underlying word produce the same hash.
   Requires a HuggingFace tokenizer at key-generation time.

3. ``char_ngram``
   Hashes character-level n-grams (stems) of the decoded context words
   instead of token IDs.  Synonym substitutions that share the same stem
   (e.g. "run" / "running" / "runs") hash to the same context, making
   the watermark invariant to morphological variation.
   Requires a HuggingFace tokenizer at key-generation time.

4. ``fixed``
   Context-independent keys generated from a seeded torch.Generator.
   Useful for testing and for eliminating hash drift entirely.
   Not robust to insertion/deletion attacks (positional mismatch).

Usage
-----
    from arcmark.side_info import SideInfoMode, compute_key_si, compute_keys_from_tokens_si

    # In encoder (processor.py):
    s_index, perm_seed = compute_key_si(
        secret_key=seed,
        context_tokens=context_tokens,
        num_keys=num_keys,
        mode=SideInfoMode.NORMALIZED,
        tokenizer=tokenizer,   # only needed for normalized / char_ngram
    )

    # In decoder (symbol_decoder.py):
    keys = compute_keys_from_tokens_si(
        secret_key=seed,
        tokens=tokens,
        context_width=context_width,
        num_keys=num_keys,
        mode=SideInfoMode.NORMALIZED,
        tokenizer=tokenizer,
    )

Both encoder and decoder must use the same mode and tokenizer for keys
to match.
"""

from __future__ import annotations

import hashlib
import re
import string
import struct
from enum import Enum
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "SideInfoMode",
    "compute_key_si",
    "compute_keys_from_tokens_si",
    "generate_fixed_key_sequence_si",
]


# ═══════════════════════════════════════════════════════════════════════════
# Mode enum
# ═══════════════════════════════════════════════════════════════════════════

class SideInfoMode(str, Enum):
    """Side-information key generation strategy."""
    HASH_CONTEXT = "hash_context"   # raw token IDs (original behaviour)
    NORMALIZED   = "normalized"     # lowercased + punctuation-stripped text
    CHAR_NGRAM   = "char_ngram"     # character n-gram stem hashing
    FIXED        = "fixed"          # context-independent (positional)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _secret_key_bytes(secret_key: int) -> bytes:
    """Encode the shared secret as a 16-byte (128-bit) little-endian block.

    The previous implementation used ``struct.pack("<q", secret_key)``, a
    *signed 64-bit* container. That capped the effective seed entropy at
    64 bits (and raised for any value that did not fit in an int64), so the
    realized key schedule could never reach the 128-bit security parameter
    (lambda = 128) claimed in the paper. Encoding to 16 bytes lets the full
    128-bit seed flow into the SHA-256 input, so ``s_index``, ``perm_seed``,
    and ``R_t`` are all derived from the complete seed.
    """
    if secret_key < 0:
        # Preserve two's-complement-style round-tripping for any legacy
        # negative seeds, still within 128 bits.
        secret_key &= (1 << 128) - 1
    return int(secret_key).to_bytes(16, "little", signed=False)


def _sha256_to_key(
    secret_key: int,
    payload: bytes,
    num_keys: int,
) -> tuple[int, int, float]:
    """Hash (secret_key || payload) and split into (s_index, perm_seed, R_t).

    The 256-bit SHA-256 digest is partitioned into three disjoint 8-byte
    blocks, mirroring the paper's key-generation equation in which the
    per-token side information is Lambda_t = F_kappa(...) split into
    (Lambda_t^{(1)}, Lambda_t^{(2)}, Lambda_t^{(3)}):

        digest[0:8]   -> s_index    (channel index k_t^{(1)}, mod num_keys)
        digest[8:16]  -> perm_seed  (vocabulary permutation seed, Lambda^{(2)})
        digest[16:24] -> R_t        (posterior-matching randomness,
                                     Rand(Lambda^{(3)}) mapped into (0, 1))

    R_t is returned as a float in the open-ish interval [0, 1); callers that
    only need the ArcMark side information can ignore it (see
    :func:`compute_key_si`, which stays backward-compatible by default).
    """
    buf = _secret_key_bytes(secret_key) + payload
    digest = hashlib.sha256(buf).digest()
    s_index   = struct.unpack("<Q", digest[0:8])[0] % num_keys
    perm_seed = struct.unpack("<Q", digest[8:16])[0]
    # Map the third 64-bit block uniformly into [0, 1). Dividing by 2**64
    # yields a value in [0, 1); this is the fixed public conversion function
    # Rand(.) of the paper.
    r_raw = struct.unpack("<Q", digest[16:24])[0]
    R_t = r_raw / float(1 << 64)
    return s_index, perm_seed, R_t


def _token_ids_to_bytes(token_ids: tuple[int, ...]) -> bytes:
    """Pack a tuple of token IDs into bytes (original behaviour)."""
    buf = b""
    for tok in token_ids:
        buf += struct.pack("<q", tok)
    return buf


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation from a text string."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _stem(word: str) -> str:
    """Very lightweight suffix-stripping stem (no external dependency).

    Strips common English suffixes so that morphological variants of the
    same root map to the same string:
        running -> run,  runs -> run,  happily -> happi,  quickly -> quick
    """
    word = word.lower()
    for suffix in ("ingly", "ingly", "tion", "ing", "ness", "ment",
                   "ful", "less", "ous", "ive", "ble", "est", "ily",
                   "ily", "ied", "ies", "ers", "er", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens_to_normalized_bytes(
    token_ids: tuple[int, ...],
    tokenizer: Any,
) -> bytes:
    """Decode token IDs to text, normalize, and encode as UTF-8 bytes."""
    if not token_ids:
        return b""
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    normalized = _normalize_text(text)
    return normalized.encode("utf-8")


def _tokens_to_char_ngram_bytes(
    token_ids: tuple[int, ...],
    tokenizer: Any,
    ngram: int = 3,
) -> bytes:
    """Decode tokens, stem each word, hash character n-grams, return bytes.

    Each word is stemmed and then represented by its first ``ngram``
    characters.  This makes morphological variants (run/running/runs)
    map to the same representation.
    """
    if not token_ids:
        return b""
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    words = _normalize_text(text).split()
    stems = [_stem(w)[:ngram] for w in words if w]
    return " ".join(stems).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_key_si(
    secret_key: int,
    context_tokens: tuple[int, ...],
    num_keys: int,
    mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
    tokenizer: Any = None,
    return_r: bool = False,
):
    """Derive (s_index, perm_seed[, R_t]) for one token position.

    Drop-in replacement for ``keygen.compute_key`` with a mode switch.

    Args:
        secret_key:     Shared secret.
        context_tokens: Preceding token IDs (left-zero-padded tuple).
        num_keys:       Side-information alphabet size r.
        mode:           Key generation strategy.
        tokenizer:      HuggingFace tokenizer — required for NORMALIZED
                        and CHAR_NGRAM modes, ignored otherwise.
        return_r:       If ``True``, also return the synchronized
                        posterior-matching randomness ``R_t`` in ``(0, 1)``,
                        derived from the third digest block (paper Eq. for
                        ``R_t = Rand(Lambda_t^{(3)})``). Defaults to ``False``
                        so existing callers that unpack ``(s_index,
                        perm_seed)`` are unaffected.

    Returns:
        ``(s_index, perm_seed)`` when ``return_r`` is ``False`` (default),
        or ``(s_index, perm_seed, R_t)`` when ``return_r`` is ``True``.
        ``s_index`` and ``perm_seed`` are plain Python ints; ``R_t`` is a
        Python float.
    """
    if mode == SideInfoMode.HASH_CONTEXT:
        payload = _token_ids_to_bytes(context_tokens)

    elif mode == SideInfoMode.NORMALIZED:
        if tokenizer is None:
            raise ValueError("tokenizer is required for SideInfoMode.NORMALIZED")
        # Filter out padding zeros before decoding
        real_tokens = tuple(t for t in context_tokens if t != 0)
        payload = _tokens_to_normalized_bytes(real_tokens, tokenizer)

    elif mode == SideInfoMode.CHAR_NGRAM:
        if tokenizer is None:
            raise ValueError("tokenizer is required for SideInfoMode.CHAR_NGRAM")
        real_tokens = tuple(t for t in context_tokens if t != 0)
        payload = _tokens_to_char_ngram_bytes(real_tokens, tokenizer)

    elif mode == SideInfoMode.FIXED:
        # Fixed mode should use generate_fixed_key_sequence_si — if called
        # here it just hashes position 0 always, which is not useful.
        # Raise to catch accidental misuse.
        raise ValueError(
            "SideInfoMode.FIXED does not use compute_key_si. "
            "Use generate_fixed_key_sequence_si instead."
        )
    else:
        raise ValueError(f"Unknown SideInfoMode: {mode}")

    s_index, perm_seed, R_t = _sha256_to_key(secret_key, payload, num_keys)
    if return_r:
        return s_index, perm_seed, R_t
    return s_index, perm_seed


def compute_keys_from_tokens_si(
    secret_key: int,
    tokens: Tensor,
    context_width: int,
    num_keys: int,
    mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
    tokenizer: Any = None,
) -> list[tuple[int, int]]:
    """Compute (s_index, perm_seed) for every position in a token sequence.

    Drop-in replacement for ``keygen.compute_keys_from_tokens`` with a
    mode switch.  Both encoder and decoder must call this with the same
    mode and tokenizer for keys to match.

    Args:
        secret_key:    Shared secret.
        tokens:        1-D LongTensor of watermarked token IDs.
        context_width: Number of preceding tokens in the context window.
        num_keys:      Side-information alphabet size r.
        mode:          Key generation strategy.
        tokenizer:     HuggingFace tokenizer (required for NORMALIZED /
                       CHAR_NGRAM modes).

    Returns:
        List of (s_index, perm_seed) tuples, one per position.
    """
    token_list = tokens.tolist()
    n          = len(token_list)
    results: list[tuple[int, int]] = []

    for t in range(n):
        raw     = token_list[max(0, t - context_width): t]
        pad_len = context_width - len(raw)
        padded  = tuple([0] * pad_len + raw)

        results.append(
            compute_key_si(
                secret_key=secret_key,
                context_tokens=padded,
                num_keys=num_keys,
                mode=mode,
                tokenizer=tokenizer,
            )
        )

    return results


def generate_fixed_key_sequence_si(
    seed: int,
    length: int,
    num_keys: int,
) -> list[tuple[int, int]]:
    """Generate a context-independent key sequence (fixed / positional mode).

    Identical to ``keygen.generate_fixed_key_sequence`` — reproduced here
    so this module is self-contained.

    Args:
        seed:     Random seed.
        length:   Number of key pairs to generate.
        num_keys: Side-information alphabet size r.

    Returns:
        List of (s_index, perm_seed) tuples.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    s_vals    = torch.randint(0, int(num_keys), (int(length),), generator=gen)
    perm_vals = torch.randint(0, 2**62,         (int(length),), generator=gen)
    return [(int(s_vals[i]), int(perm_vals[i])) for i in range(int(length))]