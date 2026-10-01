# Localizing Input Uncertainty Quantification for Large Language Models via Shapley Values

---

## 🎉 Accepted at NeurIPS 2026 🎉

---

This repository contains the official PyTorch implementation of our NeurIPS 2026 paper,\
*Localizing Input Uncertainty Quantification for Large Language Models via Shapley Values*.

**Seongjun Lee‡ · Suwan Yoon‡ · Changhee Lee\***\
Korea University\
‡ Co-First Authors\
\* Corresponding Author

## Installation

```bash
conda env create -f environment.yml
conda activate uq
```

Python 3.10 is required.

## Prepare AmbigQA

Download and validate the official AmbigQA development split:

```bash
python semantic_uncertainty5/data/prepare_ambigqa.py
```

The dataset is saved to:

```text
semantic_uncertainty5/data/ambig_qa/dev_light.json
```

To use an existing archive or a different output path:

```bash
python semantic_uncertainty5/data/prepare_ambigqa.py \
  --archive /path/to/ambignq_light.zip \
  --output /path/to/dev_light.json
```

## Set the OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

The key is read only from the environment and is not written to the saved run configuration.

## Run a low-cost sample

```bash
cd semantic_uncertainty5

python run.py \
  --query "What is the bank?" \
  --model openai/gpt-4 \
  --m 2 \
  --n 1 \
  --max_new_tokens_answer 32
```

To bypass automatic localization and force a span during a smoke test:

```bash
python run.py \
  --query "Who played the lead in Titanic?" \
  --forced_span "lead::role or chemical element" \
  --model openai/gpt-4 \
  --m 2 \
  --n 1 \
  --max_new_tokens_answer 32
```

## Run the AmbigQA experiment

From `semantic_uncertainty5`:

```bash
bash experiments/run_openrouter_experiment_gpt4.sh
```

The script uses 200 randomly selected AmbigQA development examples with seed 42, three premises per localized span, five answers per condition, and five dataset workers.

The default 200-example run can issue many paid API requests. Use `MAX_SAMPLES=1` or `MAX_SAMPLES=10` first to verify your environment and account configuration.

The following environment variables can be used to override the defaults:

```bash
DATASET_PATH=/path/to/dev_light.json \
MAX_SAMPLES=10 \
MAX_WORKERS=2 \
PYTHON_BIN=python \
bash experiments/run_openrouter_experiment_gpt4.sh
```

## Outputs

Dataset runs are saved under:

```text
semantic_uncertainty5/results/ambigqa/<run>/
```

The main files are:

- `results.jsonl`: per-example ShaQ results
- `metrics_total_aleatoric.json`: total ShaQ attribution metrics
- `metrics_max_span_phi.json`: maximum-span attribution metrics
- `metrics.json`: max-span metrics alias
- `config.yaml`: saved run configuration without the API key
- `logs/run.log`: run log
- `logs/trace.jsonl`: answer and clustering traces
- `code_snapshot.tar.gz`: source snapshot for the run

Check whether any dataset examples failed before using the metrics:

```bash
grep -H '"status": "error"' results/ambigqa/*/results.jsonl
```

No output means that no error records were found.
