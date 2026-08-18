# Vendored from the official StealthInk implementation:
#   https://github.com/yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models
#   (StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models,
#    Jiang, Wu, Kordi Boroujeny, Mark, Zeng; ICML 2025.)
# Classes/functions below are copied verbatim from
# StealthInk/every_step_1_24_direct_detect.py (module-level script parts and
# argparse removed; imports adapted to package-relative). The decode replay
# loop, which lives inline inside that script's generate(), is re-exposed
# here as stealthink_decode() with identical logic.
import math
import random

import numpy as np
import torch
from scipy.stats import norm
from transformers import LogitsProcessor

from .every_step_processor import WatermarkBase

class ReweightProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reweight(self, seed, original_token_probs, pos_embedded_message, base): 
        """
        implementing reweight function as fig. 2 in paper
        """
        self._seed_rng(seed)
        vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng).detach().cpu().tolist()
        colorlist = torch.chunk(torch.tensor(vocab_perm), base)
        original_probs_tensor = torch.tensor([original_token_probs[tok] for tok in vocab_perm], dtype=torch.float64)

        red_tokens_alpha = 0
        red_tokens_beta = 0
        for i in range(base):
            if i < pos_embedded_message:
                red_tokens_alpha += len(colorlist[i])
            if i == pos_embedded_message:
                red_tokens_beta = red_tokens_alpha + len(colorlist[i])

        if red_tokens_alpha == 0:
            alpha = torch.tensor(0.0, dtype=torch.float64)
        else:
            alpha = original_probs_tensor.cumsum(dim=0)[red_tokens_alpha - 1]
        beta = original_probs_tensor.cumsum(dim=0)[red_tokens_beta - 1]

        acc = torch.zeros_like(original_probs_tensor, dtype=torch.float64)
        acc += original_probs_tensor.cumsum(dim=0)
        acc = torch.cat((torch.tensor([0.0], dtype=torch.float64), acc))

        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b, 2 * acc - 1,
                        torch.where(
                            acc <= c, acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )
            else:  # beta <= 0.5, 0 p 2p
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b,
                        2 * acc - 1,
                        torch.where(
                            acc <= c,
                            acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )

        reweighted_probs = mapped[1:] - mapped[:-1]
        combined = {k: v for k, v in zip(vocab_perm, reweighted_probs)}
        sorted_vals = torch.tensor([combined[k] for k in sorted(combined.keys())], dtype=torch.float64)
        v_non_zero = torch.where(sorted_vals > 0, sorted_vals, torch.tensor(1e-50, dtype=torch.float64))
        logits = torch.log(v_non_zero).to(dtype=torch.float32)
        return logits


class ReweightLogitsProcessor(LogitsProcessor):
    def __init__(self, reweight_processor, embedded_message, n_gram_len, R, converted_msg_length, seen_seeds=None, cache_max=50000):
        super().__init__()
        self.reweight_processor = reweight_processor
        self.n_gram_len = n_gram_len
        self.base = int(1 / R)
        self.converted_msg_length = converted_msg_length
        self.embedded_message = embedded_message
        self.output_logits = None

        self.seen_seeds = seen_seeds if seen_seeds is not None else set()
        self.is_r = False

        # cache: seed_tuple -> (vocab_perm (cpu tensor), colorlist_indices)
        self._perm_cache = {}
        self._cache_max = cache_max

    def _get_perm_and_chunks(self, seed, base, vocab_size):
        seed_tuple = tuple(seed.view(-1).tolist())
        hit = self._perm_cache.get(seed_tuple)
        if hit is not None:
            return hit

        self.reweight_processor._seed_rng(seed)
        vocab_perm = torch.randperm(vocab_size, device='cpu', generator=self.reweight_processor.rng)
        colorlist = torch.chunk(vocab_perm, base)

        if len(self._perm_cache) >= self._cache_max:
            self._perm_cache.clear()
        self._perm_cache[seed_tuple] = (vocab_perm, colorlist)
        return self._perm_cache[seed_tuple]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        logits = scores
        seed = input_ids[:, -self.n_gram_len:]  # [1, H]
        seed_tuple = tuple(seed.view(-1).tolist())

        # skip repeated seed (purely in-memory; no disk I/O)
        if seed_tuple in self.seen_seeds:
            # print("repeated!")
            self.is_r = True
            self.output_logits = logits
            return logits
        self.seen_seeds.add(seed_tuple)
        self.is_r = False

        # bit position from CPU RNG (deterministic)
        self.reweight_processor._seed_rng(seed)
        bit_pos = torch.randint(low=0, high=self.converted_msg_length, size=(1,), generator=self.reweight_processor.rng).item()
        # print("no r, bit_pos in generation:", bit_pos)
        pos_embedded_message = self.embedded_message[bit_pos]

        # probs on the same device
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [V] on device

        # get vocab perm on CPU, then index on device with a moved view
        vocab_size = probs.shape[-1]
        vocab_perm_cpu, colorlist = self._get_perm_and_chunks(seed, self.base, vocab_size)
        vocab_perm = vocab_perm_cpu.to(device)

        # reorder probs by permutation (vectorized)
        original_probs_tensor = probs.index_select(dim=0, index=vocab_perm).to(torch.float64)

        # compute alpha/beta via cumsum (vectorized)
        cdf = original_probs_tensor.cumsum(dim=0)
        chunk_sizes = [len(t) for t in colorlist]
        red_alpha = sum(chunk_sizes[:pos_embedded_message])
        red_beta  = red_alpha + chunk_sizes[pos_embedded_message]

        alpha = cdf[red_alpha - 1] if red_alpha > 0 else torch.tensor(0.0, dtype=torch.float64, device=device)
        beta  = cdf[red_beta - 1]

        # build acc = [0, cdf]
        acc = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cdf], dim=0)

        # piecewise mapping (same logic, fully tensorized on device)
        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))
            else:  # beta <= 0.5 (0 p 2p)
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))

        reweighted_probs = (z[1:] - z[:-1]).clamp_min(1e-50)  # avoid log(0)
        # map back to original vocab order
        logits_out = torch.full_like(probs, fill_value=-1e9, dtype=torch.float32)
        logits_out.index_copy_(0, vocab_perm, reweighted_probs.log().to(torch.float32))
        self.output_logits = logits_out

        return logits_out.unsqueeze(0)  # [1, V]


class DetectorProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_colorlist_ids(self, input_ids: torch.LongTensor, base) -> torch.LongTensor:
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng)
        colorlist = torch.chunk(vocab_perm, base)
        return colorlist


def _compute_norm_p_val(cl_total, R):
    T_total = 0
    t_total = 0
    min_p_value = 10.0
    msg = []

    for _, value in cl_total.items():
        T = sum(value)
        if T:
            t = min(value)
            cur_msg = [i for i, v in enumerate(value) if v == t]
            msg.append(cur_msg)
            z = (t - R * T) / (math.sqrt(R * (1 - R) * T))
            cur_p_value = 1 - pow((1 - norm.cdf(z)), len(value))
            if cur_p_value < min_p_value:
                min_p_value = cur_p_value
            T_total += T
            t_total += t
        else:
            cur_msg = [int(random.choice(np.arange(len(value))))]
            msg.append(cur_msg)

    p_value = norm.cdf((t_total - R * T_total) / (math.sqrt(R * (1 - R) * T_total))) if T_total > 0 else 0.5
    return p_value, msg


# -------------------------
# Exact-length generation
# -------------------------
@torch.no_grad()
def generate_exact_n_tokens(
    model,
    tokenizer,
    inputs,                   # [1, prompt_len] on correct device
    logits_processor,  # includes watermark processor
    n_new_tokens=300,
    do_sample=True,
    temperature=1.0,
    top_k=0,           # 0 = no truncation
    eos_id=None,       # optional: for soft penalty only
    soft_eos_penalty: float = 0.0,  # e.g., 5.0; 0.0 disables
):
    device = inputs.device
    seq = inputs.clone()
    cur_input_ids = inputs
    past_key_values = None

    for step in range(n_new_tokens):
        out = model(input_ids=cur_input_ids, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :]   # [1, V]
        past_key_values = out.past_key_values

        # Apply processors (need full seq as input_ids for seed-based logic)
        logits = logits_processor(seq, logits)

        # Optional soft penalty on EOS before last step
        if soft_eos_penalty > 0.0 and eos_id is not None and step < n_new_tokens - 1:
            logits[:, eos_id] = logits[:, eos_id] - soft_eos_penalty

        # Sampling or greedy
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float('-inf'))
                filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = torch.softmax(filtered, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]

        seq = torch.cat([seq, next_token], dim=1)
        cur_input_ids = next_token  # only feed the last token next

    return seq  # [1, prompt_len + n_new_tokens]


# ---------------------------------------------------------------------------
# Detection replay, identical to the inline loop in the official script's
# generate() ("Detection replay (watermarked)" block), exposed as a function.
# Returns (cl_total, p_value, msg) where msg is the per-position list of
# argmin-count candidates exactly as produced by _compute_norm_p_val.
# ---------------------------------------------------------------------------
def stealthink_decode(full_ids, prompt_len, generation_length, n_gram_len,
                      processor, detector_processor, converted_msg_length,
                      num_value, R):
    gen_ids = full_ids[0, -generation_length:].tolist()
    cl_total = {i: [0 for _ in range(num_value)] for i in range(converted_msg_length)}
    detector_cache = {}  # seed_tuple -> colorlist

    for step in range(generation_length):
        if step < n_gram_len:
            continue
        # slice seed in full sequence: [prompt + step - H : prompt + step]
        l = prompt_len + step - n_gram_len
        r = prompt_len + step
        cur_seed = full_ids[:, l:r]  # [1, H]
        seed_tuple = tuple(cur_seed[0].tolist())

        # deterministic bit position
        processor._seed_rng(cur_seed)
        bit_position = torch.randint(low=0, high=converted_msg_length, size=(1,),
                                     generator=processor.rng).item()

        # colorlist cache
        hit = detector_cache.get(seed_tuple)
        if hit is None:
            detector_processor._seed_rng(cur_seed)
            vocab_perm = torch.randperm(detector_processor.vocab_size, device='cpu',
                                        generator=detector_processor.rng)
            colorlist = torch.chunk(vocab_perm, num_value)
            detector_cache[seed_tuple] = colorlist
        else:
            colorlist = hit
        new_token = gen_ids[step]
        for guessed_info in range(num_value):
            if new_token in colorlist[guessed_info]:
                cl_total[bit_position][guessed_info] += 1

    p_value, msg = _compute_norm_p_val(cl_total, R)
    return cl_total, p_value, msg
