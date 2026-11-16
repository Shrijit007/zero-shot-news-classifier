# Zero-Shot News Classifier

A Python project demonstrating zero-shot text classification with the
`facebook/bart-large-mnli` checkpoint. It classifies an English passage against
candidate labels supplied at prediction time; it does not fine-tune the model
on those application labels.

## Features

- NLI-based zero-shot classification with a shared `NewsClassifier` backend.
- Single-label ranking (`multi_label=False`) and independent multi-label scores
  (`multi_label=True`).
- Streamlit browser interface.
- Command-line predictions with optional JSON export.
- Twelve-example demonstration evaluation.
- Input validation, token-length checks, model revision metadata and inference timing.
- Seven lightweight software tests that do not download model weights.

## 1. Requirements

- Python 3.10+
- Internet access for the first model download from Hugging Face.
- CPU is supported by default. A compatible GPU can be configured separately.

The model checkpoint is downloaded and cached by Hugging Face Transformers on
first use. Model weights are not included in this repository.

## 2. Installation

```bash
git clone https://github.com/shrijit007/zero-shot-news-classifier.git
cd zero-shot-news-classifier

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only PyTorch installation, the `requirements-cpu.txt` file is provided.
If your platform needs a specific PyTorch wheel, follow the official PyTorch
installation selector and install the matching build before the other packages.

## 3. Run the browser application

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Enter a passage,
provide at least two candidate labels, choose single-label or multi-label mode,
and submit the form.

## 4. Run from the command line

```bash
python cli.py "The team won the final cricket match."
```

Specify custom labels:

```bash
python cli.py "The team won the final cricket match." \
  --labels sports technology politics education
```

Use independent multi-label scoring:

```bash
python cli.py "The government announced a new education funding policy." \
  --labels sports politics education --multi-label
```

Save the result as JSON:

```bash
python cli.py "The team won the final cricket match." \
  --labels sports technology politics education \
  --output result.json
```

Pin a checkpoint revision:

```bash
python cli.py "A sample news passage." --revision d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
```

## 5. Run the demonstration evaluation

```bash
python evaluate.py
```

The script classifies twelve hand-written passages (three per topic) and writes
the prediction records to `examples/sample_results.json`. This is a small
demonstration set, not an independent benchmark. Do not generalize its result
to arbitrary news data.

## 6. Run software tests

```bash
python -m unittest discover -s tests -v
```

The tests use mocked model inference so they do not download model weights.
They verify input handling, label validation, result alignment and CLI
argument parsing. Model-quality evaluation is a separate activity.

## 7. Project structure

```text
zero-shot-news-classifier/
├── classifier.py
├── app.py
├── cli.py
├── evaluate.py
├── requirements.txt
├── requirements-cpu.txt
├── .gitignore
├── examples/
│   └── demo_passages.json
└── tests/
    └── test_classifier.py
```

## 8. Scoring interpretation

- In single-label mode, candidate entailment logits are normalized across the
  supplied labels. The scores sum to approximately one and depend on the
  candidate set.
- In multi-label mode, each label is scored independently against contradiction.
  These scores do not need to sum to one.
- Neither score type is automatically a calibrated probability of correctness.
  The project does not apply a multi-label selection threshold.

## 9. Reproducibility

Record the exact checkpoint revision, hypothesis template, candidate labels,
scoring mode, library versions, hardware and evaluation date. Use the same
passages and label policy when comparing model configurations.

## 10. Limitations and privacy

The demonstration focuses on short English passages and topic labels. It does
not implement long-document aggregation, multilingual evaluation, automatic
label-set quality detection or a calibrated rejection policy. Hosted services
are not used: inference runs locally after the checkpoint is downloaded.
Avoid treating classification as fact verification; a false news passage can
still be classified by its topic.

## 11. License and model terms

This repository is an educational implementation. The model checkpoint and
its dependencies are governed by their respective licenses and terms. Review
the model card at https://huggingface.co/facebook/bart-large-mnli before use.
