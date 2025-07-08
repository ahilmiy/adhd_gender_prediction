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
    # X: sadece connectome verisi
    X = connectome_df.drop(columns=["participant_id"])

    # y: sadece etiketler
    y = labels_df.copy()

    # Kolonları normalize et
    y.columns = [col.strip().upper().replace(" ", "_") for col in y.columns]

    # Örneğin:
    # participant_id | ADHD_Outcome | Sex_F gibi hale getir
    if "SEX_F" not in y.columns:
        if "SEX" in y.columns:
            y["SEX_F"] = y["SEX"].map({"F": 1, "M": 0})

    return X, y[["ADHD_OUTCOME", "SEX_F"]]
def load_test_data():
    test_connectome = pd.read_csv("D:/Dosyalar/adhd_gender_prediction/data/raw/test/test_functional_connectome_matrices.csv", index_col=0)
    test_categorical = pd.read_excel("D:/Dosyalar/adhd_gender_prediction/data/raw/test/test_categorical.xlsx")

    # Gerekirse ID'leri aynı formatta ayarla
    test_connectome.index.name = "participant_id"
    return test_connectome, test_categorical


