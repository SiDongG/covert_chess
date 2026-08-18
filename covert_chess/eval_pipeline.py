#!/usr/bin/env python3
"""
File 2 of 2 — blind, judge, and score in one run.

Takes transcripts_chess.json (from generate_transcripts.py) and:
  1. BLINDS it into an A/B preference set (neutral speaker names, no labels,
     watermarked side randomized per item, item order shuffled). Writes
     blinded_pairs.jsonl + answer_key.jsonl to the output dir.
  2. JUDGES each pair with the Gemini API (one call per pair, temperature 0,
     resumable — already-judged item_ids are skipped so a crash/rate-limit
     stall doesn't cost you the whole run). Writes judge_results.jsonl.
     Optionally runs a swapped second pass (position-bias control).
  3. SCORES the verdicts against the answer key: per-setting and overall
     watermarked win-rate (excluding ties) with an exact two-sided binomial
     test vs p=0.5. If a swapped pass was run, also reports swap consistency
     and a position-bias-controlled score.

The blinded_pairs.jsonl is written to disk BEFORE any API call on purpose: it
is the resumable checkpoint. The answer key is kept separate and is never shown
to the judge.

Setup:
    export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
    pip install google-genai

Usage:
    # full pipeline, single pass:
    python eval_pipeline.py transcripts_chess.json outdir

    # with swapped-position consistency check:
    python eval_pipeline.py transcripts_chess.json outdir --swap-check

    # only score already-collected results (no API calls):
    python eval_pipeline.py transcripts_chess.json outdir --score-only

    # only blind (produce the JSONLs, don't call the API):
    python eval_pipeline.py transcripts_chess.json outdir --blind-only
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
from math import comb
from collections import Counter


# ===========================================================================
# 1. BLINDING
# ===========================================================================
def render(transcript) -> str:
    """Turns -> plain text with neutral speaker names. Strips 'kind' and stats
    so nothing leaks which condition produced the transcript."""
    lines = []
    for t in transcript["turns"]:
        speaker = f"Speaker {t['agent'] + 1}"
        lines.append(f"{speaker}: {t['content']}")
    return "\n".join(lines)


def blind(data, seed):
    """Build (blinded_pairs, answer_key) lists from transcripts_chess.json."""
    rng = random.Random(seed)
    settings = data["meta"]["settings"]
    pairs_out, key_out = [], []

    for setting in settings:
        for pair in data[setting]:
            item_id = f"{setting}-{pair['pair_id']:03d}"
            wm_text = render(pair["watermarked"])
            cl_text = render(pair["clean"])

            wm_slot = rng.choice(["A", "B"])
            a_text, b_text = ((wm_text, cl_text) if wm_slot == "A"
                              else (cl_text, wm_text))

            pairs_out.append({"item_id": item_id,
                              "transcript_A": a_text,
                              "transcript_B": b_text})
            key_out.append({"item_id": item_id,
                            "setting": setting,
                            "pair_id": pair["pair_id"],
                            "watermarked_slot": wm_slot})

    # Shuffle presentation order so settings aren't judged in blocks.
    order = list(range(len(pairs_out)))
    rng.shuffle(order)
    pairs_out = [pairs_out[i] for i in order]
    key_out.sort(key=lambda r: r["item_id"])  # order irrelevant for the key
    return pairs_out, key_out


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ===========================================================================
# 2. JUDGING (Gemini)
# ===========================================================================
SYSTEM_PROMPT = """You are evaluating the quality of two conversation transcripts. You will see
Transcript A and Transcript B. They are two versions of a conversation on the
same topic between the same speakers.

Judge them ONLY on conversational quality: fluency, naturalness, coherence
across turns, and how well each response fits what came before. Ignore length
differences and ignore which opinions or facts are expressed.

Respond with ONLY a JSON object, no markdown fences, no explanation outside
the JSON, in exactly this format:

{"item_id": "<the item_id you were given>", "preferred": "A" | "B" | "tie", "confidence": "low" | "medium" | "high", "reason": "<one sentence>"}

Rules:
- "preferred" must be exactly "A", "B", or "tie".
- Use "tie" only if the two are genuinely indistinguishable in quality.
- Output nothing before or after the JSON object."""

USER_TEMPLATE = """item_id: {item_id}

=== Transcript A ===
{a}

=== Transcript B ===
{b}

Which transcript is higher quality?"""

FLIP = {"A": "B", "B": "A", "tie": "tie"}


def judge_one(client, gtypes, model, item, swap):
    a, b = item["transcript_A"], item["transcript_B"]
    if swap:
        a, b = b, a
    msg = USER_TEMPLATE.format(item_id=item["item_id"], a=a, b=b)
    resp = client.models.generate_content(
        model=model,
        contents=msg,
        config=gtypes.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    text = resp.text.strip()
    if text.startswith("```"):                # belt and braces
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    out = json.loads(text)

    pref = str(out.get("preferred", "")).strip()
    if pref not in ("A", "B", "tie"):
        raise ValueError(f"bad 'preferred' value: {pref!r}")
    if swap:
        pref = FLIP[pref]                      # back to the original frame
    return {"item_id": item["item_id"],       # trust our id, not the echo
            "preferred": pref,
            "confidence": out.get("confidence", ""),
            "reason": out.get("reason", ""),
            "swapped_presentation": swap}


def judge_all(pairs_path, out_path, model, swap, sleep, max_retries):
    """Resumable: skips item_ids already present in out_path."""
    from google import genai
    from google.genai import types as gtypes

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
    client = genai.Client(api_key=api_key)

    items = read_jsonl(pairs_path)

    done = set()
    if os.path.exists(out_path):
        for row in read_jsonl(out_path):
            done.add(row.get("item_id"))
        if done:
            print(f"  resuming: {len(done)} items already judged in "
                  f"{os.path.basename(out_path)}")

    todo = [it for it in items if it["item_id"] not in done]
    print(f"  {len(todo)} items to judge with {model} (swap={swap}).")

    with open(out_path, "a") as out_fh:
        for i, item in enumerate(todo, 1):
            for attempt in range(max_retries):
                try:
                    row = judge_one(client, gtypes, model, item, swap)
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    break
                except Exception as e:
                    wait = min(2 ** attempt * 2, 60)
                    print(f"  [{item['item_id']}] attempt {attempt+1} failed: {e} "
                          f"-- retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
            else:
                print(f"  [{item['item_id']}] GAVE UP after {max_retries} attempts",
                      file=sys.stderr)
            if i % 25 == 0:
                print(f"    {i}/{len(todo)} done")
            time.sleep(sleep)


# ===========================================================================
# 3. SCORING
# ===========================================================================
def verdict(pref, wm_slot):
    if pref == "tie":
        return "tie"
    return "wm" if pref == wm_slot else "clean"


def binom_p_two_sided(k, n):
    """Exact two-sided binomial test vs p=0.5, no scipy needed."""
    if n == 0:
        return float("nan")
    p_obs = comb(n, k) / 2 ** n
    p = sum(comb(n, i) for i in range(n + 1)
            if comb(n, i) / 2 ** n <= p_obs + 1e-12) / 2 ** n
    return min(1.0, p)


def report(tag, rows):
    settings = sorted({r["setting"] for r in rows})
    print(f"\n== {tag} ==")
    for s in settings + ["ALL"]:
        sub = rows if s == "ALL" else [r for r in rows if r["setting"] == s]
        c = Counter(r["verdict"] for r in sub)
        wm, cl, tie = c["wm"], c["clean"], c["tie"]
        n_dec = wm + cl
        rate = wm / n_dec if n_dec else float("nan")
        p = binom_p_two_sided(wm, n_dec)
        print(f"{s:>7}: wm {wm:3d}  clean {cl:3d}  tie {tie:3d}  "
              f"| wm win-rate (ex ties) {rate:.3f}  binom p={p:.3f}")


def score(key, results_path, swapped_path=None):
    key_by_id = {r["item_id"]: r for r in read_jsonl(key)} if isinstance(key, str) \
        else {r["item_id"]: r for r in key}
    res = {r["item_id"]: r for r in read_jsonl(results_path)}

    rows = []
    for iid, r in res.items():
        if iid not in key_by_id:
            print(f"  warning: {iid} in results but not in key; skipping",
                  file=sys.stderr)
            continue
        k = key_by_id[iid]
        rows.append({"item_id": iid, "setting": k["setting"],
                     "verdict": verdict(r["preferred"], k["watermarked_slot"])})
    report("Pass 1 (all judged items)", rows)

    if swapped_path and os.path.exists(swapped_path):
        res2 = {r["item_id"]: r for r in read_jsonl(swapped_path)}
        both = sorted(set(res) & set(res2))
        agree = {iid for iid in both
                 if res[iid]["preferred"] == res2[iid]["preferred"]}
        disagree = set(both) - agree
        print(f"\nSwap consistency: {len(agree)}/{len(both)} agree "
              f"({100*len(agree)/max(len(both),1):.1f}%); "
              f"{len(disagree)} position-sensitive items")

        cons = [r for r in rows if r["item_id"] in agree]
        report("Consistent items only (position-bias controlled)", cons)

        # position-sensitive items counted as ties
        merged = [dict(r, verdict="tie") if r["item_id"] in disagree else r
                  for r in rows if r["item_id"] in set(both)]
        report("Both passes, inconsistent counted as tie", merged)


# ===========================================================================
# Driver
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="transcripts_chess.json")
    ap.add_argument("outdir", help="directory for blinded/key/results files")
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed for the blinding (default 1234)")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="e.g. gemini-2.5-flash or gemini-2.5-pro")
    ap.add_argument("--swap-check", action="store_true",
                    help="also run a swapped-position second pass and report bias")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between API calls (raise on rate limits)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--blind-only", action="store_true",
                    help="write blinded_pairs/answer_key and stop (no API calls)")
    ap.add_argument("--score-only", action="store_true",
                    help="skip blinding + judging; just score existing results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pairs_path = os.path.join(args.outdir, "blinded_pairs.jsonl")
    key_path = os.path.join(args.outdir, "answer_key.jsonl")
    res_path = os.path.join(args.outdir, "judge_results.jsonl")
    res_swap_path = os.path.join(args.outdir, "judge_results_swapped.jsonl")

    # --- 1. blind (unless we're only scoring) ---
    if not args.score_only:
        with open(args.infile) as fh:
            data = json.load(fh)
        pairs_out, key_out = blind(data, args.seed)
        write_jsonl(pairs_path, pairs_out)
        write_jsonl(key_path, key_out)
        print(f"Blinded {len(pairs_out)} pairs -> {pairs_path}")
        print(f"Answer key -> {key_path}")
        if args.blind_only:
            return

    # --- 2. judge ---
    if not args.score_only:
        print("\nJudging (pass 1)...")
        judge_all(pairs_path, res_path, args.model, swap=False,
                  sleep=args.sleep, max_retries=args.max_retries)
        if args.swap_check:
            print("\nJudging (swapped pass)...")
            judge_all(pairs_path, res_swap_path, args.model, swap=True,
                      sleep=args.sleep, max_retries=args.max_retries)

    # --- 3. score ---
    if not os.path.exists(key_path):
        sys.exit(f"No answer key at {key_path}; run without --score-only first.")
    if not os.path.exists(res_path):
        sys.exit(f"No judge results at {res_path}; run the judging step first.")
    swapped = res_swap_path if os.path.exists(res_swap_path) else None
    score(key_path, res_path, swapped)


if __name__ == "__main__":
    main()