import pandas as pd
import numpy as np
from joblib import load

from data_pipeline import load_test_data

def main():
    # 1. Test verisini yükle
    print("📂 Loading test data...")
    connectome_test_df, _ = load_test_data()

    participant_ids = connectome_test_df.index.tolist()

    # 2. ADHD modelini yükle ve tahmin yap
    print("🤖 Predicting ADHD_Outcome...")
    adhd_model = load("D:/Dosyalar/adhd_gender_prediction/model/rf_adhd_pipeline.joblib")
    adhd_preds = adhd_model.predict(connectome_test_df)

    # 3. SEX_F modelini yükle ve tahmin yap
    print("🤖 Predicting SEX_F...")
    sex_model_bundle = load("D:/Dosyalar/adhd_gender_prediction/model/lgbm_sex_regressor.joblib")
    sex_model = sex_model_bundle["model"]
    sex_threshold = sex_model_bundle["threshold"]
    sex_probs = sex_model.predict(connectome_test_df)
    sex_preds = (sex_probs >= sex_threshold).astype(int)

    # 4. DataFrame oluştur
    submission_df = pd.DataFrame({
        "participant_id": participant_ids,
        "ADHD_Outcome": adhd_preds.astype(int),
        "Sex_F": sex_preds.astype(int)
    })

    # 5. Kaydet
    output_path = "D:/Dosyalar/adhd_gender_prediction/results/submission.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Kaggle submission dosyası kaydedildi: {output_path}")

if __name__ == "__main__":
    main()
