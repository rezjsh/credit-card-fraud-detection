# Credit Card Fraud Detection

An end-to-end, modular MLOps pipeline for detecting credit card fraud on the
[Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
(V1–V28 PCA features + Amount, binary `Class` label), plus a Streamlit app for
live predictions, batch scoring, SHAP explainability, and a model performance
dashboard.

## Project Highlights

- **Architecture**: Strategy + Factory patterns across every stage (ingestion,
  validation, transformation, training, evaluation), each behind an abstract
  interface so components are swappable without touching the orchestrators.
- **Imbalance-aware evaluation**: AUPRC, F2, recall-floor thresholding, and a
  configurable financial cost model (FP/FN cost in EUR) instead of relying on
  accuracy.
- **Explainability**: SHAP-based feature importance both offline
  (`components/model_evaluation/plots.py`) and interactively in the app.
- **Serving**: A Streamlit app that reproduces the exact
  feature-engineering → scaling → feature-selection chain used in training.

## Project Structure

```
configs/                    # config.yaml, params.yaml, schema.yaml
src/credit_card_fraud_detection/
  components/
    data_ingestion/         # local CSV / Kaggle API strategies
    data_eda/                # analyzers + visualizers
    data_validation/         # pluggable validator registry
    data_transformation/     # clean → engineer → scale → select → split → resample
    model_trainer/           # model factory, hyperparameter tuner
    model_evaluation/        # metrics, threshold optimizer, calibration, plots
  config/configuration.py    # ConfigurationManager (yaml -> typed configs)
  entity/config_entity.py    # dataclasses for every stage's config
  pipeline/                  # stage_0N_*.py orchestrators, run from main.py
  utils/                     # logging, common IO helpers
app/
  app.py                     # Streamlit UI
  inference.py               # FraudInferencePipeline (loads artifacts, scores data)
  explainability.py          # SHAP helpers
main.py                      # runs the full pipeline stage-by-stage
```

## Setup

```bash
uv venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e .                    # or: pip install -e .
cp .env.example .env                   # then set KAGGLE_API_TOKEN if using kaggle_api ingestion
```

Requires Python 3.12+ (see `.python-version`).

## Running the pipeline

Edit `configs/config.yaml` / `configs/params.yaml` as needed, then either run
stages individually or via `main.py` (uncomment the stages you want — they're
commented out by default except ingestion and evaluation):

```bash
python main.py
```

Or run a single stage directly, e.g.:

```bash
python -m credit_card_fraud_detection.pipeline.stage_04_data_transformation
python -m credit_card_fraud_detection.pipeline.stage_05_model_trainer
python -m credit_card_fraud_detection.pipeline.stage_06_model_evaluation
```

This produces, under `artifacts/`:

| Path                                                               | Produced by |
| ------------------------------------------------------------------ | ----------- |
| `data_transformation/artefacts/scaler.pkl`                         | Stage 4     |
| `data_transformation/artefacts/feature_selector.pkl`               | Stage 4     |
| `data_transformation/splits/{train,val,test}.parquet`              | Stage 4     |
| `model_trainer/models/*.pkl`, `best_model.pkl`                     | Stage 5     |
| `model_evaluation/evaluation_report.json`, `model_comparison.json` | Stage 6     |
| `model_evaluation/plots/*.png`                                     | Stage 6     |

## Running the Streamlit app

Once the artifacts above exist:

```bash
streamlit run app/app.py
```

Pages:

- **Overview** — pipeline config + which model is currently loaded.
- **Predict — Single Transaction** — manual V1–V28 + Amount entry (or load a
  random real row from the test split), live fraud probability, SHAP
  explanation for that row.
- **Predict — Batch (CSV)** — upload a raw-schema CSV, get scored/labeled
  rows back, download the result; shows AUPRC/recall/precision if a `Class`
  column is present.
- **Explainability** — global SHAP feature importance over a sample.
- **Model Performance Dashboard** — renders `evaluation_report.json`,
  `model_comparison.json`, and the saved evaluation plots.

### Known limitation

`FeatureEngineer` computes `Amount_zscore` from the training split's
mean/std but doesn't persist those two numbers. `app/inference.py`
approximates them from the full raw dataset on first run and caches the
result to `artifacts/data_transformation/artefacts/inference_amount_stats.json`.
This is a close proxy, not a bit-identical reproduction of the training-time
statistic. For exact reproduction, persist `engineer.amount_stats` at the end
of `stage_04_data_transformation.py` and load it in `inference.py` instead.

## Testing

```bash
pytest --cov=src
```

## License

MIT — see `LICENSE`.