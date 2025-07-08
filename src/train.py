import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import sys
import os

# Proje ana klasörünü sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import load_data, prepare_dataset


def train_and_evaluate():
    # Veriyi yükle
    connectome_df, categorical_df, labels_df = load_data()
    X, y = prepare_dataset(connectome_df, categorical_df, labels_df)

    # Eğitim ve test setine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli tanımla
    base_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model = MultiOutputClassifier(base_model)

    # Eğit
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Raporla
    print("ADHD_Outcome sonuçları:")
    print(classification_report(y_test.iloc[:, 0], y_pred[:, 0]))

    print("\nSex_F sonuçları:")
    print(classification_report(y_test.iloc[:, 1], y_pred[:, 1]))

    # Kaydet
    dump(model, "D:/Dosyalar/adhd_gender_prediction/model/random_forest_multi.joblib")
    print("\n✅ Model 'model/random_forest_multi.joblib' olarak kaydedildi.")


if __name__ == "__main__":
    train_and_evaluate()
