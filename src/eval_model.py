import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import load

# src içindeki modülleri ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline import load_data, prepare_dataset

# Model yolları
MODELS = {
    "adhd_outcome": "D:/Dosyalar/adhd_gender_prediction/model/rf_adhd_smote.joblib",
    "sex_f": "D:/Dosyalar/adhd_gender_prediction/model/rf_sex_smote.joblib"
}

def evaluate_model(target):
    print(f"\n=== Evaluation for: {target.upper()} ===")
    
    # Veriyi hazırla
    connectome_df, categorical_df, labels_df = load_data()
    X, y_all = prepare_dataset(connectome_df, categorical_df, labels_df)
    y = y_all[[target]]

    # Train-test ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modeli yükle
    model = load(MODELS[target])

    # Accuracy (eğitim ve test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")

    # Test seti tahmini
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{target.upper()} - Confusion Matrix")
    plt.show()

    # Cross-Validation
    scores = cross_val_score(model, X, y.values.ravel(), cv=5, scoring='f1')
    print("Cross-Validation F1 Scores:", scores)
    print(f"Mean F1 Score: {np.mean(scores):.3f}")

if __name__ == "__main__":
    evaluate_model("adhd_outcome")
    evaluate_model("sex_f")
