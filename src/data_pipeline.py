import pandas as pd
import numpy as np

def load_raw_data():
    phenotypes = pd.read_csv("data/raw/phenotypic_data.csv")
    connectome = np.load("data/raw/connectome_data.npy")
    targets = pd.read_csv("data/raw/train.csv")
    return phenotypes, connectome, targets

def preprocess_data(phenotypes, connectome):
    # Imputer, encoder, flatten connectome
    # Return: X_df, y_df
    pass
