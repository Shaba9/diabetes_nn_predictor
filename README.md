# Diabetes Neural Network Lab (CSV → Train/Predict)

This project trains a small neural network to predict diabetes risk from physiological biomarkers in a CSV file (e.g., Pima Indians Diabetes dataset) and can later **predict** on a new CSV using the saved model.

> **Disclaimer (educational):** This tool is for a class/lab demonstration only. It is **not** a medical device and must not be used for diagnosis.

## Project Structure

- `diabetes_nn_tool.py` — main CLI tool (train & predict)
- `requirements.txt` — Python dependencies
- `outputs/` — created automatically (runs, reports, plots, predictions)

## Quick Start

### 1) Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train
Put your dataset CSV in this folder (example: `diabetes.csv`) and run:
```bash
python diabetes_nn_tool.py --mode train --csv diabetes.csv --target Outcome
```

This creates a timestamped run folder under `outputs/` containing:
- `diabetes_model.keras` — trained neural network
- `preprocessor.joblib` — fitted preprocessing pipeline
- `metadata.json` — schema info (feature columns, target, etc.)
- `report.md` / `report.json` — human-friendly summary
- `predictions.csv` — predictions for the full dataset
- `training_curves.png` / `confusion_matrix.png`

### 4) Predict on a new CSV
Use the run folder created during training (example shown) and predict:
```bash
python diabetes_nn_tool.py --mode predict --csv new_patients.csv --model_dir outputs/run_20260201_103500
```

The tool writes:
- `outputs/predict_YYYYMMDD_HHMMSS/predictions.csv`
- `outputs/predict_YYYYMMDD_HHMMSS/report.md`

If the new CSV **also** includes the target column (e.g., `Outcome`), the tool will compute metrics too.

## Dataset Expectations

- One row per patient.
- Biomarker columns as features.
- A target column for training (default name in many datasets is `Outcome`).

Typical Pima dataset columns:
- `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

## Tips

- If your target column name differs, pass `--target YOUR_TARGET_COL` in train mode.
- For best results, keep the same feature columns between training and prediction.

