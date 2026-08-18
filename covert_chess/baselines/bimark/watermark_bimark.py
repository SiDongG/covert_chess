# Vendored verbatim from WatermarkBimark.py of the official BiMark
# implementation: https://github.com/Kx-Feng/BiMark
# Only changes: package-relative prf import; removed the per-call
# "time cost" print (timing is handled by the harness).
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
import torch
from typing import Dict, List, Union
import numpy as np
from .prf import prf
import pandas as pd
import random
import time 

class WatermarkBimark(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        device,
        vocab_size: int,
        top_k: int = 50,
        partition_seeds: list = list(range(10)), 
        c_key: int = 530773,  
        bit_idx_key: list = 283519,
        delta: float = 1.0,
        window_size: int = 2,
        bits: str = '0',
        alpha: int = 1
    ):  
        
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.device = device
        self.top_k = top_k

        self.partition_masks = []
        if type(partition_seeds) is not list:
            partition_seeds = [partition_seeds]
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * 0.5)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            mask = torch.tensor(mask).to(torch.bool).to(device)
            self.partition_masks.append(mask)
        self.c_key = c_key
        self.bit_idx_key = bit_idx_key

        self.delta = delta
        self.alpha = alpha
        self.window_size = window_size
        self.cnt = 0
        self.bits = bits

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        begin = time.time()
        if self.cnt == 0:
            self.hist = [set() for _ in range(input_ids.shape[0])]

        if self.cnt < self.window_size:
            self.cnt += 1
            return scores

        # original probability distribution
        score_topk = torch.topk(scores, self.top_k, dim=-1)
        prob_topk_values = torch.nn.functional.softmax(score_topk.values, dim=-1).to(self.device).to(torch.float32)
        prob_topk_indices = score_topk.indices

        self.cnt += 1

        prob_delta = torch.full((input_ids.shape[0], 1), self.delta, dtype=torch.float32).to(self.device)
        alpha = torch.full((input_ids.shape[0], 1), self.alpha, dtype=torch.float32).to(self.device)
        
        
        inputs = input_ids.to(self.device)
        prefix = inputs[:, -self.window_size:]
        
        c_seed=prf(prefix, self.c_key) # seeds for bit_flip
        bit_idx_seed = prf(prefix, self.bit_idx_key) # seeds for message bit index

        # record the signs of delta (the signs of beta are the opposite of delta) 
        # which depends on the message bit XOR balance bit   
        ops_stack = []
        skip_pos = []  # record the locations of repeated seeds
       
        for i in range(prefix.size(0)):
            if prefix[i] in self.hist[i]:  # do not use repeated seeds for watermarking
                skip_pos.append(i)
            else:
                self.hist[i].add(prefix[i])

            rng_c = np.random.default_rng(c_seed[i])
            c_list = rng_c.integers(0, 2, size=len(self.partition_masks))

            rng_bit_idx = np.random.default_rng(bit_idx_seed[i])
            bit_idx = rng_bit_idx.integers(0, len(self.bits))
            bit = int(self.bits[bit_idx])
            
            ops_list = []
            for c in c_list:
                if (c == 1 and bit == 0) or (c == 0 and bit == 1):
                    ops_list.append(1)
                elif (c == 0 and bit == 0) or (c == 1 and bit == 1):
                    ops_list.append(-1)
            ops_stack.append(ops_list)
        

        ops_stack = torch.tensor(ops_stack).to(self.device)
        
        for i in range(len(self.partition_masks)): 
            top_k_mask = self.partition_masks[i][prob_topk_indices]
            p0 = torch.sum(prob_topk_values * top_k_mask, -1, keepdim=True)
            mask_p0 = (p0 < 1e-30) + (1 - p0 < 1e-30)

            # values of delta and beta dependen on the input distribution
            delta = torch.max(torch.min(alpha / p0, 1 + prob_delta), torch.ones(prob_delta.shape).to(self.device)) - 1
            beta = torch.min(delta * p0 / (1 - p0), torch.ones(prob_delta.shape).to(self.device))

            delta[mask_p0 == 1] = 0
            beta[mask_p0 == 1] = 0

            delta[skip_pos] = 0
            beta[skip_pos] = 0

            delta = delta * ops_stack[:,i].unsqueeze(1)
            beta = beta * ops_stack[:,i].unsqueeze(1)

            delta = delta.expand(-1, prob_topk_values.shape[1])
            beta = beta.expand(-1, prob_topk_values.shape[1])

            prob_topk_values[top_k_mask == True] = prob_topk_values[top_k_mask == True] * (1+delta)[top_k_mask == True]
            prob_topk_values[top_k_mask == False] = prob_topk_values[top_k_mask == False] * (1-beta)[top_k_mask == False]

        prob = torch.zeros_like(scores, dtype=torch.float32).to(self.device)
        prob.scatter_(1, prob_topk_indices, prob_topk_values)
        
        new_scores = torch.log(prob).to(self.device)

        return new_scores