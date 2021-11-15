import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from census_class import CensusData

# Create the app object
app = FastAPI()
pickle_in = open("model/model.pkl", "rb")
classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict")
def predict_salary(data: CensusData):
    data = data.dict()
    age = data["age"]
    fnlgt = data["fnlgt"]
    education_num = data["education_num"]
    capital_gain = data["capital_gain"]
    capital_loss = data["capital_loss"]
    hours_per_week = data["hours_per_week"]
    x3_Female = data["x3_Female"]
    x3_Male = data["x3_Male"]

    prediction = classifier.predict(
        [[age, fnlgt, education_num, capital_gain, capital_loss, hours_per_week, x3_Female, x3_Male]]
    )
    if prediction[0] > 0.5:
        prediction = ">50k"
    else:
        prediction = "<=50k"
    return {"prediction": prediction}
