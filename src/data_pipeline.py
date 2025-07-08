import pandas as pd


def load_data():
    """
    train_new klasöründen verileri yükler:
    - connectome verisi (flatten edilmiş)
    - anket verisi (kategorik)
    - hedef değişkenler (cinsiyet, adhd)
    """
    base_path = "D:/Dosyalar/adhd_gender_prediction/data/raw/train_new/"

    # Connectome matrisleri (flatten edilmiş)
    connectome_df = pd.read_csv(base_path + "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")

    # Kategorik metadata (anket/survey verileri)
    categorical_df = pd.read_excel(base_path + "TRAIN_CATEGORICAL_METADATA_new.xlsx")

    # Hedef değişkenler
    labels_df = pd.read_excel(base_path + "TRAINING_SOLUTIONS.xlsx")

    return connectome_df, categorical_df, labels_df


from sklearn.impute import SimpleImputer

def prepare_dataset(connectome_df, categorical_df, labels_df):
    labels_df.columns = labels_df.columns.str.lower()

    df = connectome_df.merge(categorical_df, on="participant_id", how="left")
    df = df.merge(labels_df, on="participant_id", how="left")

    # Hedef değişkenler
    y = df[["adhd_outcome", "sex_f"]].astype(int)

    # Özellikler
    drop_cols = ["participant_id", "adhd_outcome", "sex_f"]
    X = df.drop(columns=drop_cols)

    # Eksik değerleri doldur (mean imputation)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y

