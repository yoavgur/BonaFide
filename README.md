# BonaFide

<!-- <p align="center">
  <img
    width="160"
    height="160"
    alt="bonafide_logo"
    src="https://github.com/user-attachments/assets/5cd12724-642b-4221-8a5b-7cb0f2837805"
    style="background-color: white;"
  />
</p> -->

<p>
  <img
    src="https://github.com/user-attachments/assets/a51b670c-b1bd-4fed-8af5-a9e04f0d4524"
    alt="BonaFide logo"
    width="160"
    align="center"
  />
Code for reproducing the results of the **BonaFide** paper on chain-of-thought
faithfulness in language models.

This repository contains the full pipeline for building BonaFide-style
labeled CoT data — generation, LLM-judge analysis, label extraction, and
dataset filtering — together with the faithfulness metric implementations
evaluated in the paper. Experimental notebooks, annotation UI, internal
analysis scripts, and the raw dataset CSVs themselves have been omitted.</p>

Code for reproducing the results of the **BonaFide** paper on chain-of-thought
faithfulness in language models.

This repository contains the full pipeline for building BonaFide-style
labeled CoT data — generation, LLM-judge analysis, label extraction, and
dataset filtering — together with the faithfulness metric implementations
evaluated in the paper. Experimental notebooks, annotation UI, internal
analysis scripts, and the raw dataset CSVs themselves have been omitted.

See the complete BonaFide dataset and leaderboard on [HuggingFace](https://huggingface.co/collections/yoavgurarieh/bonafide), and the paper on [arXiv](https://arxiv.org/pdf/2605.25052)!

<img width="1255" height="648" alt="image" src="https://github.com/user-attachments/assets/ae942550-a79f-485e-9cd1-de9368666338" />

## Layout

```
.
├── env.py                  # Reads GEMINI_API_KEY
├── isolate_steps.py        # Judge (Gemini wrapper), JSON parser, entailment retrieval
├── sentence_splitting.py   # Lightweight quote-aware sentence splitter
├── multipass_judge.py      # Multi-pass LLM-judge pipeline (parallel tracks + post-processing)
├── multipass_prompts.py    # Prompt templates for the multi-pass pipeline
├── extract_labels.py       # Turn judge PipelineResult objects into step/CoT-level label rows
├── filter_labels.py        # Dedupe + balance the merged labels CSV into the final dataset
├── metrics/                # Faithfulness metric suite
│   ├── adding_mistakes/    │
│   ├── cc_shap/            │
│   ├── early_answering/    │
│   ├── filler_tokens/      │  Each metric implements the
│   ├── fur/                │  FaithfulnessMetric interface in
│   ├── monitor/            │  metrics/base.py.
│   ├── paraphrasing/       │
│   ├── regex_baseline/     │
│   ├── scm/                │
│   └── simulatability/     │
└── generation/             # Reproducible LLM generation pipeline (vLLM / HF backends)
```

## Setup

Install dependencies (PyTorch, transformers, sentence-transformers, vLLM,
google-genai, pandas, scipy, tqdm) into a Python 3.10+ environment, then set
the required credentials:

```bash
export GEMINI_API_KEY=...   # used by the Judge (Gemini API)
export HF_TOKEN=...         # (optional) HuggingFace token, picked up automatically for gated model downloads
```

## Reproducing results

### 1. Generate CoT rollouts

```bash
python -m generation --model <hf-model-id> --input questions.csv --output rollouts.csv
```

See `python -m generation --help` for the full option list (backend choice,
sampling parameters, thinking-tag handling, resume support, etc.).

### 2. Run the multi-pass LLM judge and extract labels

The `multipass_judge` module runs the per-CoT judge pipeline; `extract_labels`
turns its `PipelineResult` objects into a labels DataFrame:

```python
import pandas as pd
from isolate_steps import Judge
from multipass_judge import run_multipass_from_thinking
from extract_labels import extract_labels

judge = Judge()  # uses GEMINI_API_KEY
rollouts = pd.read_csv("rollouts.csv").to_dict("records")

items = []
for row in rollouts:
    gt_steps = str(row["steps"]).splitlines() if row.get("steps") else None
    result = run_multipass_from_thinking(
        question=row["prompt"],
        hint=row.get("prompted_hint", ""),
        thinking=row["cot"],
        model_answer=row.get("model_answer", ""),
        judge=judge,
        ground_truth_steps=gt_steps,
    )
    items.append((result, row))

labels_df = extract_labels(items)
labels_df.to_csv("labels_prefilter.csv", index=False)
```

See `multipass_judge.run_multipass_from_thinking` for the full set of judge
options (chunk size, adversarial validation, dual-adversarial, parallelism,
etc.).

### 3. Filter into the final labeled dataset

```bash
python filter_labels.py \
    --input labels_prefilter.csv \
    --output labeled_dataset.csv \
    --max-per-group 100
```

Dedupes identical spans, caps each `(target_model, label_type)` group, and
balances across source task types (hinting / complex / graph / hinted_complex).

### 4. Run a faithfulness metric

```bash
python -m metrics --model <hf-model-id> \
    --input labeled_dataset.csv \
    --output metric_results.csv \
    --metric early_answering
```

Available `--metric` values: `adding_mistakes`, `cc_shap`, `early_answering`,
`filler_tokens`, `fur`, `paraphrasing`, `regex_baseline`, `scm`,
`simulatability`, `monitor` (plus the `monitor_no_hint`, `monitor_generic`,
`monitor_no_tool` variants). See `python -m metrics --help` for per-metric
flags.

## Datasets

The CSV inputs used in the paper are **not** bundled with this repository, but can be found on [HuggingFace](https://huggingface.co/collections/yoavgurarieh/bonafide).
The pipeline expects standard CSV inputs (one prompt per row, plus the
columns referenced in each metric's `--help` output).

---

## Citation

```bibtex
@misc{gurarieh2026faithfulnessmetricsdontmeasure,
      title={Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth}, 
      author={Yoav Gur-Arieh and Ana Marasović and Mor Geva},
      year={2026},
      eprint={2605.25052},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.25052}, 
}
```
