import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import dump

# src klasörünü tanıt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import load_data, prepare_dataset
from src.feature_engineering.meta_features import extract_meta_features  # meta feature'ları dahil et

def train_and_save_model(X, y, target_name, model_path):
    print(f"\n=== Training model for: {target_name} ===")

    pipeline = make_pipeline(
        SMOTE(random_state=42),
        RandomForestClassifier(random_state=42)
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y.values.ravel(), cv=skf, scoring='f1')

    print("Cross-Val F1 Scores:", scores)
    print(f"Mean F1: {np.mean(scores):.3f}")

    # Tüm veri ile eğit ve kaydet
    pipeline.fit(X, y.values.ravel())
    dump(pipeline, model_path)
    print(f"✅ Model kaydedildi: {model_path}")

def main():
    # Veriyi yükle
    connectome_df, categorical_df, labels_df = load_data()
    _, y_all = prepare_dataset(connectome_df, categorical_df, labels_df)

    # === ADHD_OUTCOME Modeli ===
    X_full, _ = prepare_dataset(connectome_df, categorical_df, labels_df)
    train_and_save_model(
        X_full,
        y_all[["ADHD_OUTCOME"]],
        target_name="ADHD_OUTCOME",
        model_path="D:/Dosyalar/adhd_gender_prediction/model/rf_adhd_pipeline.joblib"
    )

    # === SEX_F Modeli (meta features + categorical) ===
    meta_features = extract_meta_features(connectome_df)
    X_sex = pd.concat([meta_features, categorical_df.reset_index(drop=True)], axis=1)
    y_sex = y_all[["SEX_F"]]

    train_and_save_model(
        X_sex,
        y_sex,
        target_name="SEX_F",
        model_path="D:/Dosyalar/adhd_gender_prediction/model/rf_sex_pipeline_meta.joblib"
    )

if __name__ == "__main__":
    main()
