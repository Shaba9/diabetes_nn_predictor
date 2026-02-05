import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------
# Utilities
# -------------------------

def make_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def risk_bucket(p: float) -> str:
    if p < 0.33:
        return 'Low'
    elif p < 0.66:
        return 'Moderate'
    return 'High'


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                fill_val = mode.iloc[0] if len(mode) else 'Unknown'
                df[col] = df[col].fillna(fill_val)
    return df


def detect_feature_types(df: pd.DataFrame, target_col: str, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop'
    )


def build_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )
    return model


def plot_training_curves(history: keras.callbacks.History, out_path: str):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], label='Train Loss')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    if 'accuracy' in hist:
        plt.plot(epochs, hist['accuracy'], label='Train Acc')
    if 'val_accuracy' in hist:
        plt.plot(epochs, hist['val_accuracy'], label='Val Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Pred 0', 'Pred 1'])
    plt.yticks(tick_marks, ['True 0', 'True 1'])

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'confusion_matrix': cm.tolist(),
    }, cm


def write_markdown_report(path: str, title: str, summary: dict, metrics: dict | None, files: list[str], disclaimer=True):
    lines = []
    lines.append(f"# {title}
")
    lines.append(f"**Run time:** {datetime.now().isoformat(timespec='seconds')}
")
    for k, v in summary.items():
        lines.append(f"- **{k}:** {v}")
    lines.append('')

    if metrics:
        lines.append('## Metrics')
        lines.append(f"- Accuracy: **{metrics['accuracy']:.3f}**")
        lines.append(f"- Precision: **{metrics['precision']:.3f}**")
        lines.append(f"- Recall: **{metrics['recall']:.3f}**")
        lines.append(f"- F1-score: **{metrics['f1']:.3f}**")
        lines.append(f"- ROC-AUC: **{metrics['roc_auc']:.3f}**" if metrics['roc_auc'] is not None else "- ROC-AUC: **N/A**")
        lines.append('')
        lines.append('### Confusion Matrix')
        lines.append('Rows = true label, columns = predicted label
')
        lines.append('```')
        lines.append(str(metrics['confusion_matrix']))
        lines.append('```
')

    lines.append('## Files Produced')
    for f in files:
        lines.append(f"- `{f}`")

    if disclaimer:
        lines.append('
---')
        lines.append('**Disclaimer:** This project is educational and not intended for medical use.')

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('
'.join(lines))


# -------------------------
# Train mode
# -------------------------

def run_train(args):
    out_root = make_dir(args.out_dir)
    run_dir = make_dir(os.path.join(out_root, f"run_{now_stamp()}"))

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV. Columns: {list(df.columns)}")

    df = fill_missing(df)

    y = df[args.target].astype(int).values
    feature_cols, numeric_cols, categorical_cols = detect_feature_types(df, args.target)
    X = df[feature_cols].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(np.unique(y)) == 2 else None
    )

    # Preprocess
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Model
    model = build_model(input_dim=X_train_p.shape[1], learning_rate=args.lr)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train_p, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks
    )

    # Evaluate
    y_prob = model.predict(X_test_p, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics, cm = compute_metrics(y_test, y_pred, y_prob)

    # Human-friendly console output
    print('
' + '=' * 72)
    print('DIABETES PREDICTION — TRAIN MODE (Educational)')
    print('=' * 72)
    print(f"Dataset: {os.path.basename(args.csv)}")
    print(f"Rows: {len(df):,} | Features: {len(feature_cols)} | Target: {args.target}")
    print('-' * 72)
    print('Test Performance:')
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1-score : {metrics['f1']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else '  ROC-AUC  : N/A')
    print('-' * 72)
    print('Confusion Matrix (rows=True label, cols=Pred label):')
    print(np.array(metrics['confusion_matrix']))

    # Save predictions for full dataset
    X_all_p = preprocessor.transform(X)
    all_prob = model.predict(X_all_p, verbose=0).ravel()
    all_pred = (all_prob >= 0.5).astype(int)

    preds_df = df.copy()
    preds_df['Predicted'] = all_pred
    preds_df['Probability'] = all_prob
    preds_df['Risk'] = [risk_bucket(p) for p in all_prob]

    preds_path = os.path.join(run_dir, 'predictions.csv')
    preds_df.to_csv(preds_path, index=False)

    # Save artifacts
    model_path = os.path.join(run_dir, 'diabetes_model.keras')
    preproc_path = os.path.join(run_dir, 'preprocessor.joblib')
    meta_path = os.path.join(run_dir, 'metadata.json')

    model.save(model_path)
    joblib.dump(preprocessor, preproc_path)

    metadata = {
        'task': 'diabetes_binary_classification',
        'target': args.target,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'threshold': 0.5,
        'train_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'test_size': args.test_size,
            'random_state': args.random_state,
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2)

    # Plots
    curves_path = os.path.join(run_dir, 'training_curves.png')
    cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    plot_training_curves(history, curves_path)
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), cm_path)

    # Reports
    report_json_path = os.path.join(run_dir, 'report.json')
    report_md_path = os.path.join(run_dir, 'report.md')

    report = {
        'run_dir': run_dir,
        'csv': args.csv,
        'rows': int(len(df)),
        'features': int(len(feature_cols)),
        'target': args.target,
        'metrics': metrics,
        'artifacts': {
            'model': os.path.basename(model_path),
            'preprocessor': os.path.basename(preproc_path),
            'metadata': os.path.basename(meta_path),
        },
        'notes': [
            'Supervised learning: model trains on labeled examples (Outcome 0/1).',
            'Weights updated via backpropagation to minimize binary cross-entropy loss.'
        ]
    }
    with open(report_json_path, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2)

    write_markdown_report(
        report_md_path,
        title='Diabetes Prediction — Neural Network Report (Train)',
        summary={
            'Dataset': os.path.basename(args.csv),
            'Rows': f"{len(df):,}",
            'Features': f"{len(feature_cols)}",
            'Target': args.target,
            'Run folder': run_dir,
        },
        metrics=metrics,
        files=[
            'predictions.csv',
            'diabetes_model.keras',
            'preprocessor.joblib',
            'metadata.json',
            'training_curves.png',
            'confusion_matrix.png',
            'report.json',
        ]
    )

    print('
Outputs saved to:', run_dir)
    print('Artifacts:')
    print(' -', model_path)
    print(' -', preproc_path)
    print(' -', meta_path)
    print(' -', preds_path)
    print(' -', report_md_path)
    print('=' * 72 + '
')


# -------------------------
# Predict mode
# -------------------------

def run_predict(args):
    if not args.model_dir:
        raise ValueError('--model_dir is required in predict mode (path to a training run folder).')

    model_dir = args.model_dir
    model_path = os.path.join(model_dir, 'diabetes_model.keras')
    preproc_path = os.path.join(model_dir, 'preprocessor.joblib')
    meta_path = os.path.join(model_dir, 'metadata.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Missing preprocessor file: {preproc_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    with open(meta_path, 'r', encoding='utf-8') as fp:
        meta = json.load(fp)

    target_col = meta['target']
    feature_cols = meta['feature_cols']
    threshold = float(meta.get('threshold', 0.5))

    df = pd.read_csv(args.csv)
    df = fill_missing(df)

    # Verify features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            'Your prediction CSV is missing feature columns used during training: ' + ', '.join(missing)
        )

    X = df[feature_cols].copy()

    preprocessor = joblib.load(preproc_path)
    model = keras.models.load_model(model_path)

    X_p = preprocessor.transform(X)
    prob = model.predict(X_p, verbose=0).ravel()
    pred = (prob >= threshold).astype(int)

    out_root = make_dir(args.out_dir)
    pred_dir = make_dir(os.path.join(out_root, f"predict_{now_stamp()}"))

    out_df = df.copy()
    out_df['Predicted'] = pred
    out_df['Probability'] = prob
    out_df['Risk'] = [risk_bucket(p) for p in prob]

    preds_path = os.path.join(pred_dir, 'predictions.csv')
    out_df.to_csv(preds_path, index=False)

    # If target exists in the prediction CSV, compute metrics
    metrics = None
    cm = None
    if target_col in df.columns:
        y_true = df[target_col].astype(int).values
        metrics, cm = compute_metrics(y_true, pred, prob)

    # Console output
    print('
' + '=' * 72)
    print('DIABETES PREDICTION — PREDICT MODE (Educational)')
    print('=' * 72)
    print(f"Input CSV: {os.path.basename(args.csv)}")
    print(f"Loaded model from: {model_dir}")
    print(f"Rows predicted: {len(df):,}")
    print('-' * 72)

    # Show a few example predictions
    preview_n = min(8, len(out_df))
    preview = out_df[['Probability', 'Risk', 'Predicted']].head(preview_n).copy()
    if target_col in out_df.columns:
        preview.insert(0, 'Actual', out_df[target_col].astype(int).head(preview_n).values)

    with pd.option_context('display.max_columns', 50, 'display.width', 160):
        print('Example Predictions (first rows):')
        print(preview.to_string(index=False))

    if metrics:
        print('-' * 72)
        print('Metrics (because target column exists in the prediction CSV):')
        print(f"  Accuracy : {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall   : {metrics['recall']:.3f}")
        print(f"  F1-score : {metrics['f1']:.3f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else '  ROC-AUC  : N/A')

    # Reports
    report_json_path = os.path.join(pred_dir, 'report.json')
    report_md_path = os.path.join(pred_dir, 'report.md')

    report = {
        'predict_dir': pred_dir,
        'input_csv': args.csv,
        'model_dir': model_dir,
        'rows': int(len(df)),
        'target_in_csv': (target_col in df.columns),
        'metrics': metrics,
        'notes': [
            'Predict mode uses the saved preprocessor + neural network from train mode.',
            'Probabilities are mapped to simple Low/Moderate/High risk buckets for readability.'
        ]
    }
    with open(report_json_path, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2)

    write_markdown_report(
        report_md_path,
        title='Diabetes Prediction — Neural Network Report (Predict)',
        summary={
            'Input CSV': os.path.basename(args.csv),
            'Model folder': model_dir,
            'Rows': f"{len(df):,}",
            'Prediction folder': pred_dir,
        },
        metrics=metrics,
        files=['predictions.csv', 'report.json'],
    )

    print('
Saved:')
    print(' -', preds_path)
    print(' -', report_md_path)
    print('=' * 72 + '
')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train or use a neural network for diabetes prediction from a CSV dataset.'
    )
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='train: fit model and save artifacts | predict: load artifacts and predict')
    parser.add_argument('--csv', required=True, help='Path to input CSV')
    parser.add_argument('--target', default='Outcome', help='Target column name (train mode). Default: Outcome')

    # Train params
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction (train mode)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # Predict params
    parser.add_argument('--model_dir', default=None, help='Folder created by train mode (contains model + preprocessor)')

    # Output
    parser.add_argument('--out_dir', default='outputs', help='Output directory')
    return parser.parse_args()


def main():
    # Reduce TF noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = parse_args()

    if args.mode == 'train':
        run_train(args)
    else:
        run_predict(args)


if __name__ == '__main__':
    main()
