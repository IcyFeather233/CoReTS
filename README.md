# CoReTS

CoReTS synthesizes repository comprehension questions from code, collects tool-assisted solutions, and prepares accepted trajectories for supervised fine-tuning. The release includes a portable pipeline, a training recipe, and 20 examples from the training corpus.

## Contents

* `src/corets/`: repository snapshots, Python evidence extraction, model calls, teacher-agent rollout, quality review, and export.
* `configs/`: generation parameters and an example repository manifest.
* `training/`: full fine-tuning script and the Hermes notice compatibility plugin.
* `data_example/questions.jsonl`: 20 generated questions with their task IDs, repositories, and commits.
* `data_example/examples.jsonl`: corresponding training trajectories in Hermes format, including questions, tool interactions, and answers.
* `data_example/example_sources.json`: source and license information for each trajectory; license texts are in `data_example/licenses/`.

## Generation

Requirements: Python 3.11 or newer with `tarfile.data_filter`, Git, Docker, and OpenAI-compatible APIs for the scanner, question author, teacher, and reviewer. The teacher uses the public SWE-QA-Pro agent pinned to commit `93ac6a4f3af3fe3f86580f62142f47e97e2cc897`.

```bash
python -m pip install -e '.[agent]'
cp .env.example .env
# Fill in the endpoints, keys, and agent repository URL, then export them.
set -a; source .env; set +a
corets setup-agent --directory runs/swe-qa-pro --context runs/docker-context
# Set all model IDs in configs/generation.json and edit the repository manifest.
corets generate --repositories configs/repositories.example.json \
  --config configs/generation.json --work runs/campaign --agent runs/swe-qa-pro
corets export --work runs/campaign --output runs/train.jsonl
```

Repository manifests specify `owner/name`, a full 40-character commit, and a `source_url`. Source snapshots are read-only during teacher rollouts. The generation configuration controls per-repository question count, concurrency, model parameters, and six review thresholds. Only a trajectory with valid tool calls, a final answer, a passing reference review, and a passing evidence-cited candidate review is exported. Reasoning content and `<think>` blocks are excluded from exported assistant messages. Intermediate events may contain model reasoning and source material: keep the work directory private and inspect artifacts before sharing.

The released pipeline uses API-based authoring and source-grounded review; the experiments also used internal authoring and validation components.

## Training

The recipe uses ms-swift 3.12.0, full-parameter fine-tuning, eight GPUs, four epochs, global batch size 16, a 49,152-token training limit, and YaRN extension to 65,536 positions. It requires a compatible CUDA/PyTorch, FlashAttention, and DeepSpeed installation; install those for the target cluster before installing the optional training dependencies.

```bash
python -m pip install -e '.[train]'
MODEL=YOUR_MODEL MODEL_TYPE=YOUR_MODEL_TYPE DATASET=/absolute/path/to/train.jsonl OUTPUT_DIR=/absolute/path/to/output bash training/train.sh
```

Use the example data for a smoke test. Set `NPROC_PER_NODE` to change the GPU count and adjust gradient accumulation to maintain the global batch size.

## Evaluation

Use the official benchmark repositories:

* [SWE-QA](https://github.com/peng-weihan/SWE-QA-Bench)
* [SWE-QA-Pro](https://github.com/TIGER-AI-Lab/SWE-QA-Pro)

See the paper and benchmark repositories for evaluation settings.

## Tests

```bash
python -m unittest discover -s tests -v
CORETS_OFFICIAL_AGENT="$PWD/runs/swe-qa-pro" python -m unittest discover -s tests -v
```

The integration test exercises the pinned agent with simulated model and tool responses. The example data include third-party code excerpts; their upstream license texts and revisions are provided alongside the data.
