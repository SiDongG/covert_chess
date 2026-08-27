# Vendored from detect_watermark_dump.py of the official BiMark implementation
#   https://github.com/Kx-Feng/BiMark
# Notes on changes:
#  * The original file defines decode_bimark_multibit_watermark twice in the
#    same class; Python keeps the second definition (fixed partition_seeds,
#    matching the generator). We vendor that operative second definition and
#    drop the shadowed first one.
#  * Debug prints removed; imports made package-relative. 
import copy
import random
from math import sqrt

import numpy as np
from scipy import stats

from .prf import prf


class WatermarkDetector:
    def __init__(self, tokenizer, vocab_size, window_size, gamma):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.gamma = gamma

    def _compute_z_score(self, observed_green_count, total_count, proportion=False):
        if not proportion:
            proportion = self.gamma
        numer = observed_green_count - proportion * total_count
        denom = sqrt(total_count * proportion * (1 - proportion))
        z = numer / denom
        return z
    
    def _compute_p_value(self, z):
        p_value = stats.norm.sf(z)
        return p_value

    def _z_test(self, COUNT):
        if len(COUNT[0]) == 1:
            observed_green_count = np.max(COUNT, axis=1)
        elif len(COUNT[0]) == 2:
            observed_green_count = np.min(np.max(COUNT, axis=1))
        total_count = np.sum(COUNT)
        score = self._compute_z_score(observed_green_count, total_count)
        p_value = self._compute_p_value(score)
        return score, p_value, observed_green_count, total_count
    
    def _zerobit_watermark_detector_stride(self, green_count, valid_tokens, stride):
        z_score_list, z_p_value_list = [], []
        stride_list = []
        for i in range(green_count.shape[0]):
            z_score = self._compute_z_score(green_count[i], valid_tokens[i])
            z_p_value = self._compute_p_value(z_score)
            z_score_list.append(z_score)
            z_p_value_list.append(z_p_value)
            stride_list.append(stride * i)
        return z_score_list, z_p_value_list, stride_list
    
    def decode_bimark_multibit_watermark(self, inputs, partition_seeds, c_key,  bit_idx_key, bits, bits_len=0, weight=0,
                               start=0, stride=50):
        if bits_len == 0:
            bits_len = len(bits)

        if weight == 0:
            weight = [1 for _ in range(len(partition_seeds))]

        stride_idx_list = [start + stride * i for i in range((len(inputs) - start)//stride +1)]       
    
        COUNTS = [[[0, 0] for _ in range(bits_len)] for _ in range(len(stride_idx_list) )]
        
        partition_masks = []
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * 0.5)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            partition_masks.append(mask)
    
        hist = set() 
        generate_counts = [0 for _ in range(len(stride_idx_list))]
        idx_s = 0
        for t in range(self.window_size, len(inputs)):
            try:
                generate_counts[idx_s] += 1
            except:
                continue
            if idx_s < len(stride_idx_list):
                if (t-self.window_size) == stride_idx_list[idx_s]:
                    idx_s += 1
                    try:
                        COUNTS[idx_s] = copy.deepcopy(COUNTS[idx_s-1])
                        generate_counts[idx_s] = generate_counts[idx_s-1]
                    except:
                        continue
            prefix = inputs[t - self.window_size: t]
            
            c_seed=prf(prefix, c_key) # seed
            # partition_idx_seed=prf(prefix, partition_idx_key)
            rng_idx_seed = prf(prefix, bit_idx_key)
            
            if prefix not in hist:  # do not watermarking the same seed
                hist.add(prefix)
            else:
                continue
            rng_c = np.random.default_rng(c_seed)
            
            rng_bit_idx = np.random.default_rng(rng_idx_seed)

            c_list = rng_c.integers(0, 2, size=len(partition_masks))


            bit_idx = rng_bit_idx.integers(0, bits_len)

            token_idx = inputs[t].item()

            for i in range(len(partition_masks)):
                mask = partition_masks[i]
                if ((c_list[i] == 1 and (mask[token_idx].item() is False)) or (c_list[i] == 0 and (mask[token_idx].item() is True))):
                    COUNTS[idx_s][bit_idx][1] += 1 * weight[i]
                elif ((c_list[i] == 1 and (mask[token_idx].item() is True)) or (c_list[i] == 0 and (mask[token_idx].item() is False))):
                    COUNTS[idx_s][bit_idx][0] += 1 * weight[i]
                else:
                    COUNTS[idx_s][bit_idx][random.randint(0, 1)] += 1 * weight[i]

        green_counts = []
        valid_counts = []
        z_scores = []
        p_values = []
        decode_bits = ['' for _ in range(len(COUNTS))]
        hit = [0 for _ in range(len(COUNTS))]
        hit_rate = []
        for i in range(len(COUNTS)):
            count = COUNTS[i]
            for j in range(len(count)):
                if count[j][0] > count[j][1]:
                    decode_bits[i] += '0'
                    if bits[j] == '0':
                        hit[i] += 1
                elif count[j][0] < count[j][1]:
                    decode_bits[i] += '1'
                    if bits[j] == '1':
                        hit[i] += 1
                else:
                    decode_bits[i] += 'x'
            hit_rate.append(hit[i]/bits_len)
            green_count = np.max(count, axis=-1).sum()
            valid_count = np.sum(count)
            z_score = self._compute_z_score(green_count, valid_count)
            p_value = self._compute_p_value(z_score)
            green_counts.append(green_count)
            valid_counts.append(valid_count)
            z_scores.append(z_score)
            p_values.append(p_value)
        return COUNTS, generate_counts, green_counts, valid_counts, z_scores, p_values, decode_bits, hit, hit_rate
    

