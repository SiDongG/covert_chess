# Artifact: Feedback Coding Enables Inference-Time Covert Agentic Communication

This repository contains the evaluation code for the paper. It reproduces the
continuous-text primitive experiments (rate–reliability, text quality, runtime,
ablations, and the information-theoretic security diagnostic) and the
end-to-end covert-agentic-conversation experiments (turn-based reliability,
throughput, efficiency, and blinded LLM-judge conversational quality).

The scheme evaluated is **BAM** (Burnashev Adaptive Posterior Matching), a
variable-length feedback code for black-box LLM steganography, built on optimal-transport coupling and benchmarked against four open-source
multi-bit watermarking baselines (Arcmark, MPAC, BiMark, StealthInk).


## Repository layout

```
.
├── README.md                     # this file
├── ENVIRONMENT.md                # hardware/software versions, models, data
├── requirements.txt              # minimum-version dependencies
├── requirements-lock.txt         # pinned reference lockfile (regenerate before submission)
├── environment.yml               # conda alternative
├── LICENSE
├── arcmark_src/
│   └── arcmark/                  # Modified ArcMark source (upports both the BAM and Arcmark architecture)
└── covert_chess/
    ├── baselines/                # MPAC / BiMark / StealthInk source
    ├── conversation_seeds.json   # 50 (topic, opener) scenarios per setting
    ├── compare.py                # BAM vs baselines on C4          -> Fig 2, Table 1
    ├── Ablation.py               # confirmation-phase ablation     -> Fig 3 (App. C.2)
    ├── Ablation_packetization.py # packetization + runtime         -> Table 3 (App. C.1)
    ├── KeyKL.py                  # seed-marginalized per-token KL  -> Fig 4 (Sec 5.2 / App. E)
    ├── Turn_based_bam_allmodels.py       # covert agentic conversation  -> Tables 2 & 4, Fig 6 (include     three model-separated files)
    ├── Generate_Turn_based_transcript.py # paired stego/clean transcripts for the judge
    └── eval_pipeline.py          # blind + judge + score quality   -> quality (w/t/l) columns, Fig 7    
└── data/                         

```
### Files not displayed above but included in the repo

--Batch files
--Modified Arcmark source files 
--Baseline source files

## Setup

```bash
# 1. Python 3.11 environment (venv or conda)
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml

# 2. Install torch matching your CUDA toolkit first (example: CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements-lock.txt                    # exact pin  (recommended)
# or: pip install -r requirements.txt                   # minimum constraints

# 4. Add arcmark_src/arcmark/ and covert_chess/baselines/ (see "What you must add")

# 5. Gated-model + Hub access
hf auth login                                   # needed for meta-llama/*
```

A CUDA GPU is required; `Turn_based_bam_allmodels.py` exits early if no CUDA
device is visible. See `ENVIRONMENT.md` for the reference hardware.


## Claim-to-experiment map

| Paper artifact                                   | Script                              | Output files |
|--------------------------------------------------|-------------------------------------|--------------|
| Fig 2, Table 1 (C4: reliability/rate/quality/time) | `compare.py`                      | `comparison_c4.{png,csv}`, `comparison_c4_perplexity.png` |
| Fig 3 (ablation, App. C.2)                        | `Ablation.py`                       | `ablation_confirmation.{png,csv}`, `ablation_confirmation_frontier.{png,pdf}` |
| Table 3 (packetization + runtime, App. C.1)       | `Ablation_packetization.py`         | `ablation_packetization.{png,csv}` |
| Fig 4 (per-token KL security, Sec 5.2 / App. E)   | `KeyKL.py`                          | `kl_security_c4*.{png,csv}` |
| Tables 2 & 4, Fig 6 (covert agentic conversation) | `Turn_based_bam_allmodels.py`       | `turnbased_bam_matrix*.{png,csv}` (per-model + combined) |
| Quality (w/t/l) columns; Fig 7 judge prompt       | `Generate_Turn_based_transcript.py` → `eval_pipeline.py` | `transcripts_<task>.json`, `blinded_pairs.jsonl`, `answer_key.jsonl`, `judge_results.jsonl` |


## Running the experiments

All commands are run from `covert_chess/`. Each script resolves the ArcMark
source via the `ARCMARK_SRC` environment variable, defaulting to `../arcmark_src`.

```bash
cd covert_chess
export ARCMARK_SRC=../arcmark_src        # or an absolute path
```

### 1. Continuous-text: BAM vs baselines (Fig 2, Table 1)

```bash
python compare.py
```

Streams C4 RealNews, runs BAM and the baselines at fixed token budgets, and
writes the comparison CSV/PNG plus a small dump of stego/base text pairs for the
optional judge. **Reproduction note:** the full operating-point sweep used for
the Fig 2 curve is defined by `L_VALUES` near the top of `compare.py`.

### 2. Confirmation-phase ablation (Fig 3)

```bash
python Ablation.py
```

Compares two-phase BAM against the one-phase and fixed-length variants and the
fixed-length ArcMark baseline, all sharing the same posterior-matching
communication phase. Produces linear- and semilog-scale frontiers.

### 3. Packetization + runtime breakdown (Table 3)

```bash
python Ablation_packetization.py
```

Embeds a fixed payload under `1×24`, `2×12`, and `3×8` packetization with a
per-component (encode / generate / decode) wall-clock breakdown.

### 4. Per-token KL security diagnostic (Fig 4)

```bash
python KeyKL.py
```

Monte-Carlo estimate of the seed-marginalized per-token KL divergence between
stego and cover distributions under the realized PRF key schedule.

### 5. Covert agentic conversation (Tables 2 & 4, Fig 6)

```bash
python Turn_based_bam_allmodels.py
```

Runs the two-way turn-based covert channel across the instruct models and covert
tasks (tic-tac-toe / chess / go19) over the two conversational covers. The
`chat` setting corresponds to the casual-chat covertext (Table 2) and the
`debate` setting to the forum-exchange covertext (Table 4), both drawn from
`conversation_seeds.json`. Writes per-model and combined matrices and example
rollouts. The largest model shards automatically; override its placement with
`DEVICE_MAP` if needed.

### 6. Conversational-quality judging (quality columns, Fig 7)

Two steps. First generate paired watermarked/clean transcripts (the pairing
shares topic, opener, and RNG seed so the judge sees only the effect of
embedding):

```bash
# The generator auto-discovers the main script via the Turn_based_bam*.py glob.
# Pin it explicitly to be safe, and pick the covert task:
MAIN_SCRIPT=Turn_based_bam_allmodels.py TASK=chess N_PAIRS=100 \
  python Generate_Turn_based_transcript.py          # -> transcripts_chess.json
```
Additionally, one may use 
```bash
MAIN_SCRIPT=Turn_based_bam_phi4_14b.py TASK=chess N_PAIRS=100 \
  python Generate_Turn_based_transcript.py          # -> transcripts_chess.json
```
to generate json files containing only transcripts produced by specified model.

Then blind, judge, and score against the answer key (resumable; the judge never
sees the answer key):

```bash
export GEMINI_API_KEY=...                            # hosted judge
python eval_pipeline.py transcripts_chess.json outdir --swap-check
# useful variants:
#   --blind-only    write blinded_pairs/answer_key, no API calls
#   --score-only    score already-collected results
#   --model gemini-3.1-pro-preview
```

The paper uses Gemini 3.1 Pro as the judge; select the model with `--model`.


## Environment variables

| Variable                    | Used by                              | Meaning / default |
|-----------------------------|--------------------------------------|-------------------|
| `ARCMARK_SRC`               | all continuous-text + turn-based     | path to `arcmark_src` (default `../arcmark_src`) |
| `DEVICE_MAP`                | `Turn_based_bam_allmodels.py`        | device map for the 30B model (default `auto`) |
| `SEEDS_PATH`                | `Turn_based_bam_allmodels.py`        | conversation seeds (default `conversation_seeds.json`) |
| `MAIN_SCRIPT`               | `Generate_Turn_based_transcript.py`  | which `Turn_based_bam*.py` to borrow definitions from |
| `TASK`, `N_PAIRS`, `ROUNDS`, `BASE_SEED`, `OUT` | `Generate_Turn_based_transcript.py` | covert task, pair count, rounds, seed, output path |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `eval_pipeline.py`           | hosted judge credential |
| `LAPLACE_B`, `LAPLACE_B_CONF` | likelihood-model scripts           | override the Laplace likelihood scale |

Several reproduction knobs are **in-file constants**, not environment variables:
`SMOKE_TEST` (a fast sanity-check mode), `N_TRIALS` / `N_DIALOGUES`, `FIXED_NS`,
`L_VALUES`, and `MODEL_NAMES`. Edit these at the top of the relevant script to
change trial counts, token budgets, operating points, or model selection. With
`SMOKE_TEST = False` (the shipped default) the scripts run the full paper
configuration.


## Notes on reproducibility

- **Trial counts / runtime.** Full runs are 100–2000 trials per setting and
  download multi-billion-parameter models; expect substantial GPU time.
  Set `SMOKE_TEST = True` in a script for a fast end-to-end check first.
- **Randomness.** Per-trial seeds are fixed in-script (the transcript generator
  pairs each watermarked run with its clean counterpart under an identical
  seed). Exact numeric reproduction still depends on model weights, library
  versions, GPU kernels, and sampling; small variation is expected.
- **Network.** First runs download models from the Hugging Face Hub and stream
  C4. For offline clusters, pre-stage weights and set `HF_HUB_OFFLINE=1`.
- **Judge.** The conversational-quality numbers depend on a hosted third-party
  model and its version at query time; the score-only mode lets you re-score
  cached verdicts deterministically.


## License

See `LICENSE`. The artifact is released to support reproducibility of the
paper's claims, not operational misuse; it uses only public models and public
or synthetic data.