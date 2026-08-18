# Vendored verbatim from utils.py of the official BiMark implementation
#   https://github.com/Kx-Feng/BiMark
#   (BiMark: Unbiased Multilayer Watermarking for Large Language Models,
#    Feng, Zhang, Zhang, Zhang, Pan; ICML 2025.)
# Extracted into its own module so we do not inherit utils.py's heavy
# dataset-loading imports (datasets/ftfy).
import hashlib

import torch


def prf(seed: torch.LongTensor, secret_key: int):
    if seed.dim() == 1:
        seed_str = ''.join(map(str, seed.tolist())) + str(secret_key)
        hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
        hash_int = int(hash_digest, 16)
        return hash_int % 2**32
    else:
        result = []
        for row in seed:
            seed_str = ''.join(map(str, row.tolist())) + str(secret_key)
            hash_digest = hashlib.sha256(seed_str.encode()).hexdigest()
            hash_int = int(hash_digest, 16)
            result.append(hash_int % 2**32)
    return result
