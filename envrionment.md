# Environment: hardware and software

This file records the environment the experiments were run in. 
## Hardware

| Component        | Specification                                   |
|------------------|-------------------------------------------------|
| GPU              | 1 × NVIDIA A100-SXM4, 80 GB                      |
| GPU count        | 1 (batch size 1)                                |
| CPU              | 4 cores                                          |
| System memory    | 64 GB                                            |
| Scheduler        | SLURM cluster                                    |

A single 80 GB GPU is sufficient for every experiment, however, hugging-face model weights need to be downloaded in scratch folders. The largest model
(`Qwen/Qwen3-30B-A3B-Instruct-2507`, ~60 GB in fp16) is loaded with
`device_map="auto"`.

## Models

Continuous-text experiments (`compare.py`, `Ablation*.py`, `KeyKL.py`) use base
models; the covert-agentic experiments (`Turn_based_bam_allmodels.py`) use
instruct models. The exact Hugging Face repositories are set at the top of each
script:

Continuous-text (base):
- `unsloth/Meta-Llama-3.1-8B`
- `unsloth/Qwen3.5-9B-Base`
- `unsloth/mistral-7b-v0.3`

Covert-agentic (instruct):
- `meta-llama/Llama-3.1-8B-Instruct`  (gated — request access on Hugging Face)
- `microsoft/phi-4`  (MIT, ungated)
- `Qwen/Qwen3-30B-A3B-Instruct-2507`  (MoE, 3B active)

Gated models require a Hugging Face account with access granted and a login
token (`huggingface-cli login`). All models are downloaded on first use; ensure
the cluster nodes have network access to the Hugging Face Hub, or pre-stage the
weights and set `HF_HOME` / `HF_HUB_OFFLINE=1`.

## Data

- **C4 RealNews**: `allenai/c4`, config `realnewslike`, streamed at runtime
  (`load_dataset(..., streaming=True)`). Requires network access on first run.
- **Conversation seeds**: `covert_chess/conversation_seeds.json` (included) —
  50 `(topic, opener)` scenarios per setting (`chat`, `debate`).

## LLM judge (conversational quality only)

`eval_pipeline.py` uses a hosted LLM as a blinded pairwise judge. The paper uses
Gemini 3.1 Pro; the script's default is `gemini-2.5-flash` and is overridable
with `--model`. Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the
environment and the `google-genai` package.