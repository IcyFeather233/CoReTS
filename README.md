# CoReTS

CoReTS synthesizes repository comprehension questions from code, collects tool-assisted solutions, and prepares accepted trajectories for supervised fine-tuning. The release includes a portable pipeline, the Qwen3-8B training recipe, and 20 real examples from the training corpus.

## Contents

* `src/corets/`: repository snapshots, Python evidence extraction, model calls, teacher-agent rollout, quality review, and export.
* `configs/`: generation parameters and an example repository manifest.
* `training/`: Qwen3-8B full fine-tuning script and the Hermes notice compatibility plugin.
* `data/examples.jsonl`: 20 training trajectories in Hermes format. `data/example_sources.json` records the task, repository, commit and upstream license for each row; license texts are in `data/licenses/`.

## Generation

Requirements: Python 3.11 or newer with `tarfile.data_filter`, Git, Docker, and OpenAI-compatible APIs for the scanner, question author, teacher, and reviewer. The teacher uses the public SWE-QA-Pro agent pinned to commit `93ac6a4f3af3fe3f86580f62142f47e97e2cc897`. Its repository is fetched separately and is not included in this repository.

```bash
python -m pip install -e '.[agent]'
cp .env.example .env
# Fill in the endpoints and keys, then export them to the shell.
set -a; source .env; set +a
corets setup-agent --directory runs/swe-qa-pro --context runs/docker-context
# Set the scanner, author, and reviewer model IDs in configs/generation.json.
corets generate --repositories configs/repositories.example.json \
  --config configs/generation.json --work runs/campaign --agent runs/swe-qa-pro
corets export --work runs/campaign --output runs/train.jsonl
```

Repository manifests specify `owner/name` and a full 40-character commit. Source snapshots are read-only during teacher rollouts. The generation configuration controls per-repository question count, concurrency, model parameters, and six review thresholds. Only a trajectory with valid tool calls, a final answer, a passing reference review, and a passing evidence-cited candidate review is exported. Reasoning content and `<think>` blocks are excluded from exported assistant messages. Intermediate events may contain model reasoning and source material: keep the work directory private and inspect artifacts before sharing.

The public pipeline is a portable reimplementation of the experiment workflow. The original campaign used additional internal authoring and native validation infrastructure; its exact prompts, validation controls, and runtime are not fully reproduced here. The example trajectories and training hyperparameters are from the actual campaign. Model endpoints and identities must be configured by the user; the code verifies the response model against the configured accepted IDs. Do not treat this release as an exact replay of the original data generation.

## Training

The recipe uses ms-swift 3.12.0, full-parameter Qwen3-8B fine-tuning, eight GPUs, four epochs, global batch size 16, a 49,152-token training limit, and YaRN extension to 65,536 positions. It requires a compatible CUDA/PyTorch, FlashAttention, and DeepSpeed installation; install those for the target cluster before installing the optional training dependencies.

```bash
python -m pip install -e '.[train]'
DATASET=/absolute/path/to/train.jsonl OUTPUT_DIR=/absolute/path/to/output bash training/train.sh
```

The example data can be used for a smoke test, but are not the full training set. `NPROC_PER_NODE` can be set for a different GPU count; doing so changes the global batch size unless gradient accumulation is adjusted. The original campaign additionally ran a longest-example GPU preflight and runtime assertions, which are not packaged in this portable training script.

## Evaluation

Use the official benchmark repositories:

* [SWE-QA](https://github.com/peng-weihan/SWE-QA-Bench)
* [SWE-QA-Pro](https://github.com/TIGER-AI-Lab/SWE-QA-Pro)

Our experiments used benchmark-specific judging and execution settings. Consult the paper and the benchmark repositories when reproducing the reported evaluation; no evaluation implementation is included here.

## Tests

```bash
python -m unittest discover -s tests -v
CORETS_OFFICIAL_AGENT="$PWD/runs/swe-qa-pro" python -m unittest discover -s tests -v
```

The optional integration test exercises the pinned agent with simulated model and tool responses; it does not require a live model or a running container. The dataset examples include excerpts of third-party code. Their respective upstream license texts and revisions are provided alongside the data.
