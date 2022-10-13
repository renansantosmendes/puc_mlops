import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


def load_models():
    model = pickle.load(open('C:\\PUC\\03 - MLOps\\Aula 02\\model.pkl', 'rb'))
    scaler = pickle.load(open('C:\\PUC\\03 - MLOps\\Aula 02\\scaler.pkl', 'rb'))
    return scaler, model

class FeatureDataInstance(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float

scaler, model = load_models()
app = FastAPI()

@app.post('/api/predict/', status_code=200)
async def predict(data: FeatureDataInstance):
    received_data = np.array([data.baseline_value,
                              data.accelerations,
                              data.fetal_movement,
                              data.uterine_contractions,
                              data.light_decelerations,
                              data.severe_decelerations,
                              data.prolongued_decelerations]).reshape(1, -1)
    print(scaler.transform(received_data))
    prediction = model.predict(scaler.transform(received_data))
    return {'y_pred': prediction[0]}

@app.get("/")
async def read_root():
    return {"Hello": "World"}
