import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import load_data, prepare_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump

def train_with_smote_sex():
    # Veriyi yükle
    connectome_df, categorical_df, labels_df = load_data()
    X, y_all = prepare_dataset(connectome_df, categorical_df, labels_df)

    # Sadece sex etiketi
    y = y_all[["sex_f"]]

    # SMOTE ile sınıf dengeleme
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Eğitim/test ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Model eğitimi
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Tahmin ve metrikler
    y_pred = model.predict(X_test)
    print("=== SEX_F (SMOTE uygulanmış) ===")
    print(classification_report(y_test, y_pred))

    # Kaydet
    dump(model, "D:/Dosyalar/adhd_gender_prediction/model/rf_sex_smote.joblib")
    print("✅ Model kaydedildi: model/rf_sex_smote.joblib")


if __name__ == "__main__":
    train_with_smote_sex()
