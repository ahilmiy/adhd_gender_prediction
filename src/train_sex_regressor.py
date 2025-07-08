import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from joblib import dump

from data_pipeline import load_data, prepare_dataset

def train_and_evaluate(X, y, model_path):
    print("=== Training LGBM Regressor for SEX_F ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    thresholds = []
    probs_all = []
    y_true_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMRegressor(objective="binary", random_state=42)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict(X_val)
        precision, recall, thres = precision_recall_curve(y_val, y_pred_proba)

        # En iyi threshold'u F1'e göre seç
        f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1s)
        best_threshold = thres[best_idx]

        y_pred_binary = (y_pred_proba >= best_threshold).astype(int)
        f1 = f1_score(y_val, y_pred_binary)

        print(f"Fold {fold+1} - F1: {f1:.4f} - Best threshold: {best_threshold:.2f}")
        f1_scores.append(f1)
        thresholds.append(best_threshold)

        probs_all.extend(y_pred_proba)
        y_true_all.extend(y_val)

    mean_f1 = np.mean(f1_scores)
    best_threshold = np.mean(thresholds)
    print(f"\n✅ Mean F1 Score: {mean_f1:.4f}")
    print(f"✅ Average Best Threshold: {best_threshold:.2f}")

    # Tüm veriyle yeniden eğit
    final_model = LGBMRegressor(objective="binary", random_state=42)
    final_model.fit(X, y)

    dump({
        "model": final_model,
        "threshold": best_threshold
    }, model_path)

    print(f"✅ Model saved to: {model_path}")

def main():
    connectome_df, categorical_df, labels_df = load_data()
    X, y_all = prepare_dataset(connectome_df, categorical_df, labels_df)
    y = y_all["SEX_F"]

    train_and_evaluate(X, y, model_path="D:/Dosyalar/adhd_gender_prediction/model/lgbm_sex_regressor.joblib")

if __name__ == "__main__":
    main()
