# Diabetes Prediction — Neural Network Report (Train)
**Run time:** 2026-02-06T17:15:07
- **Dataset:** sample_diabetes.csv
- **Rows:** 120
- **Features:** 8
- **Target:** Outcome
- **Run folder:** outputs\run_20260206_171459

## Metrics
- Accuracy: **0.583**
- Precision: **0.619**
- Recall: **0.867**
- F1-score: **0.722**
- ROC-AUC: **0.496**

### Confusion Matrix
Rows = true label, columns = predicted label
```
[[1, 8], [2, 13]]
```
## Files Produced
- `predictions.csv`
- `diabetes_model.keras`
- `preprocessor.joblib`
- `metadata.json`
- `training_curves.png`
- `confusion_matrix.png`
- `report.json`
---
**Disclaimer:** This project is educational and not intended for medical use.