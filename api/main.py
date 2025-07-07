from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_model, predict

app = FastAPI()
model = load_model()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_ad(input_data: InputData):
    result = predict(model, input_data.features)
    return {"Sex_F": result[0], "ADHD_Outcome": result[1]}
