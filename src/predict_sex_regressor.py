import pandas as pd
import numpy as np
from joblib import load

from data_pipeline import load_test_data

def predict_sex():
    print("🔍 Loading model and threshold...")
    model_bundle = load("D:/Dosyalar/adhd_gender_prediction/model/lgbm_sex_regressor.joblib")
    model = model_bundle["model"]
    threshold = model_bundle["threshold"]
    print(f"✅ Loaded model with threshold: {threshold:.2f}")

    print("📂 Loading test data...")
    connectome_test_df, categorical_test_df = load_test_data()

    print("📊 Predicting probabilities...")
    y_pred_probs = model.predict(connectome_test_df)

    print("🧮 Applying threshold to get final predictions...")
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    results_df = pd.DataFrame({
        "participant_id": connectome_test_df.index,
        "predicted_sex_f": y_pred_binary
    })

    output_path = "D:/Dosyalar/adhd_gender_prediction/results/sex_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    predict_sex()
