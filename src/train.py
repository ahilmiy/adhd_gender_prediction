from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from joblib import dump

def train_model(X_train, y_train):
    base = RandomForestClassifier(class_weight='balanced')
    model = MultiOutputClassifier(base)
    model.fit(X_train, y_train)
    dump(model, "model/rf_multioutput.joblib")
    return model
