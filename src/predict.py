import os
import sys
import pandas as pd
from joblib import load

# src klasöründen import yapabilmek için path ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import load_data, prepare_dataset

# Model yolları
ADHD_MODEL_PATH = "D:/Dosyalar/adhd_gender_prediction/model/rf_adhd_smote.joblib"
SEX_MODEL_PATH = "D:/Dosyalar/adhd_gender_prediction/model/rf_sex_smote.joblib"

def run_prediction():
    # Veriyi yükle
    connectome_df, categorical_df, labels_df = load_data()
    X, y = prepare_dataset(connectome_df, categorical_df, labels_df)

    # Model dosyalarını yükle
    adhd_model = load(ADHD_MODEL_PATH)
    sex_model = load(SEX_MODEL_PATH)

    # Tahminleri al
    adhd_preds = adhd_model.predict(X)
    sex_preds = sex_model.predict(X)

    # Sonuçları DataFrame olarak birleştir
    results = pd.DataFrame({
        "participant_id": labels_df["participant_id"],
        "predicted_adhd": adhd_preds,
        "predicted_sex_f": sex_preds
    })

    # İlk birkaç tahmini göster
    print(results.head())

    # CSV olarak kaydet
    results.to_csv("D:/Dosyalar/adhd_gender_prediction/results/predictions.csv", index=False)
    print("✅ Tahminler kaydedildi: results/predictions.csv")


if __name__ == "__main__":
    run_prediction()
