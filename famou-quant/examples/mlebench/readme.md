# MLEBench Examples

Famou 2.0 examples for [MLEBench](https://github.com/openai/mle-bench) Kaggle competition tasks.

## Prerequisites

### Required Packages

```bash
pip install mlebench numpy pandas pynvml
```

- `mlebench` — MLEBench evaluation framework (provides the `mlebench grade-sample` CLI for scoring submissions)
- `numpy` — numerical computation
- `pandas` — data manipulation (submission CSV parsing)
- `pynvml` — NVIDIA GPU monitoring during training

### Competition Data

Each competition requires its prepared dataset. The default data directory is configured in `evaluator.py`. Download and prepare data following the [MLEBench documentation](https://github.com/openai/mle-bench).

## Usage

1. Configure your LLM credentials in the competition's `config.yaml`
2. Run:

```bash
sh examples/mlebench/<competition-name>/run.sh
```

## Available Competitions

| Competition | Task Type | Data Type |
|---|---|---|
| denoising-dirty-documents | Regression | Image |
| new-york-city-taxi-fare-prediction | Regression | Tabular |
| ventilator-pressure-prediction | Regression | Time Series |
| siim-isic-melanoma-classification | Classification | Image |
| siim-covid19-detection | Detection | Image |
| nomad2018-predict-transparent-conductors | Regression | Tabular |
| h-and-m-personalized-fashion-recommendations | Recommendation | Tabular |
