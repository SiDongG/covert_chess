#!/usr/bin/env python3
"""
For each of N pairs per setting it produces:
  * a watermarked dialogue (payload embedded via BAM), and
  * a clean dialogue (identical topic/opener seed + identical RNG seed, ordinary
    sampling, no payload).
"""
from __future__ import annotations

import os
import gc
import json
import time
import types
import glob

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Locate and load DEFINITIONS ONLY from the main experiment script.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUN_MARKER = "for model_name in MODEL_NAMES:"


def _resolve_main_script() -> str:
    """Find the experiment script whose DEFINITIONS we borrow.

    Priority: $MAIN_SCRIPT if set, else the Qwen3-30B script (the default model
    for this pipeline), else the Llama-8B script, else any Turn_based_bam*.py
    next to this file. Set MAIN_SCRIPT explicitly to switch models, e.g.
        MAIN_SCRIPT=Turn_based_bam_Llama8B.py python generate_transcripts.py
    """
    env = os.environ.get("MAIN_SCRIPT")
    if env:
        if not os.path.exists(env):
            raise SystemExit(f"MAIN_SCRIPT={env!r} does not exist.")
        return env
    preferred = [
        os.path.join(_THIS_DIR, "Turn_based_bam_Qwen3_30B.py"),
        os.path.join(_THIS_DIR, "Turn_based_bam_Llama8B.py"),
    ]
    for c in preferred:
        if os.path.exists(c):
            return c
    fallback = sorted(glob.glob(os.path.join(_THIS_DIR, "Turn_based_bam*.py")))
    if fallback:
        return fallback[0]
    raise SystemExit(
        "Could not find an experiment script. Put Turn_based_bam_Qwen3_30B.py "
        "(or Turn_based_bam_Llama8B.py) next to this file, or set "
        "MAIN_SCRIPT=/path/to/it.py"
    )


def _load_defs(path: str, marker: str) -> types.ModuleType:
    src = open(path).read()
    if marker not in src:
        raise SystemExit(
            f"Run marker {marker!r} not found in {path}. The experiment script's "
            "run section may have changed; update _RUN_MARKER."
        )
    cut = src.index(marker)
    # trim back to the start of the enclosing comment banner if present
    banner = src.rfind("# " + "=" * 76, 0, cut)
    if banner != -1:
        cut = banner
    mod = types.ModuleType("bam_defs")
    mod.__file__ = path
    mod.__dict__["__name__"] = "bam_defs"
    exec(compile(src[:cut], path, "exec"), mod.__dict__)
    return mod


_MAIN_PATH = _resolve_main_script()
M = _load_defs(_MAIN_PATH, _RUN_MARKER)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_PAIRS = int(os.environ.get("N_PAIRS", "100"))
ROUNDS = int(os.environ.get("ROUNDS", str(M.ROUNDS_PER_DIALOGUE)))
TASK_NAME = os.environ.get("TASK", "chess")
SETTINGS = ["chat", "debate"]
OUT_PATH = os.environ.get("OUT", f"transcripts_{TASK_NAME}.json")
BASE_SEED = int(os.environ.get("BASE_SEED", "70000"))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Clean (unwatermarked) dialogue: same prompts/seed, ordinary sampling.
# ---------------------------------------------------------------------------
def run_clean_dialogue(profile, seed, rounds):
    """Generate a dialogue with NO payload and NO watermark steering. Uses the
    same system prompt, opener and turn budget as the watermarked path so the
    only difference a judge sees is the watermarking itself."""
    topic = seed["topic"]
    opener = seed["opener"]
    history = [{"agent": 0, "content": opener, "kind": "opener"}]
    history_tokens = M.CTX.tokenizer.encode(opener, add_special_tokens=False)
    total_tokens = 0

    first_speaker = 1
    for turn_idx in range(rounds):
        agent = (first_speaker + turn_idx) % 2
        prompt_ids = M.CTX.render_dialogue(history, agent, profile, topic)
        toks = M.run_filler_turn(prompt_ids, history_tokens,
                                 profile.max_turn_tokens)
        total_tokens += len(toks)
        history.append({"agent": agent,
                        "content": M.decode_turn_text(toks) or "(…)",
                        "kind": "clean"})
    return {"turns": history,
            "n_turns": len(history),
            "total_tokens": int(total_tokens)}


def strip_transcript(history, total_tokens):
    """Keep only what a judge needs; drop payload/decoder metadata so the judge
    cannot tell which condition it is looking at."""
    turns = [{"agent": t["agent"],
              "content": t["content"],
              "kind": t.get("kind", "")}
             for t in history]
    return {"turns": turns,
            "n_turns": len(turns),
            "total_tokens": int(total_tokens)}


def main():
    log(f"Collecting {N_PAIRS} paired transcripts per setting "
        f"(task={TASK_NAME}, rounds={ROUNDS})")
    log(f"Experiment script: {_MAIN_PATH}")
    log(f"Model: {M.MODEL_NAMES[0]}")

    M.CTX = M.LMContext(M.MODEL_NAMES[0])
    out = {"meta": {"task": TASK_NAME,
                    "n_pairs_per_setting": N_PAIRS,
                    "rounds_per_dialogue": ROUNDS,
                    "model": M.MODEL_NAMES[0],
                    "settings": SETTINGS,
                    "note": ("watermarked and clean share the same topic/opener "
                             "seed and RNG seed per pair_id")}}
    try:
        for conv_name in SETTINGS:
            profile = M.CONVERSATIONS[conv_name]
            pairs = []
            log(f"\n=== {conv_name} (max_turn={profile.max_turn_tokens}) ===")
            for i in range(N_PAIRS):
                seed = M.pick_seed(profile, i)
                t0 = time.time()

                # --- watermarked ---
                rng = np.random.RandomState(BASE_SEED + i)
                np.random.seed(BASE_SEED + i)
                torch.manual_seed(BASE_SEED + i)
                task_gen = M.TASKS[TASK_NAME]()
                wm = M.run_dialogue(rng, task_gen, profile, seed)
                wm_t = strip_transcript(
                    wm["history"],
                    wm["wm_tokens_total"] + wm["pad_tokens_total"]
                    + wm["filler_tokens_total"])

                # --- clean (same seed, same prompts) ---
                np.random.seed(BASE_SEED + i)
                torch.manual_seed(BASE_SEED + i)
                cl = run_clean_dialogue(profile, seed, ROUNDS)
                cl_t = strip_transcript(cl["turns"], cl["total_tokens"])

                pairs.append({
                    "pair_id": i,
                    "seed": {"topic": seed["topic"], "opener": seed["opener"]},
                    "watermarked": wm_t,
                    "clean": cl_t,
                    "_wm_stats": {
                        "completed_payloads": wm["completed_payloads"],
                        "correct": wm["correct"],
                        "errors": wm["errors"],
                    },
                })
                if (i + 1) % 10 == 0 or i == 0:
                    log(f"  pair {i+1:>3}/{N_PAIRS}  "
                        f"wm_turns={wm_t['n_turns']} clean_turns={cl_t['n_turns']}  "
                        f"[{time.time()-t0:.1f}s]")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out[conv_name] = pairs
    finally:
        M.CTX.teardown()
        M.CTX = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\nWrote {OUT_PATH}")
    for s in SETTINGS:
        log(f"  {s}: {len(out[s])} pairs")


if __name__ == "__main__":
    main()