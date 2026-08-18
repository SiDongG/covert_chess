"""Vendored open-source multi-bit watermarking baselines.

- mpac:       Yoo et al., "Advancing Beyond Identification: Multi-bit
              Watermark for Large Language Models", NAACL 2024.
              https://github.com/bangawayoo/mb-lm-watermarking  (Apache-2.0)
- bimark:     Feng et al., "BiMark: Unbiased Multilayer Watermarking for
              Large Language Models", ICML 2025.
              https://github.com/Kx-Feng/BiMark  (no license file; vendored
              for research benchmarking with attribution)
- stealthink: Jiang et al., "StealthInk: A Multi-bit and Stealthy Watermark
              for Large Language Models", ICML 2025.
              https://github.com/yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models
              (no license file; vendored for research benchmarking with
              attribution)

Files are copied verbatim except where noted in per-file headers (import
paths, removed debug prints, and extraction of inline script logic into
importable functions).
"""
