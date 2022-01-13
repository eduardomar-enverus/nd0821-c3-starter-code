import pandas as pd
from fastapi import FastAPI, Body
import pickle
from census_class import CensusData
import os

from project.ml.data import process_data, categorical_features
from project.ml.model import inference

keep_cat = ["marital-status", "race", "relationship", "sex"]  # Limit sparsity


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def load_pickle(file):
    with open(file, "rb") as f:
        pickle_object = pickle.load(f)
    return pickle_object


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/model.pkl")
encoder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model/encoder.pkl"
)
lb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/labeler.pkl")

classifier = load_pickle(model_path)
encoder = load_pickle(encoder_path)
labeler = load_pickle(lb_path)

# Create the app object
app = FastAPI()


@app.get("/")
async def say_hello():
    return "Hello World!"


@app.post("/predict_salary/")
def predict_salary(
    data: CensusData = Body(
        None,
        example={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
):

    data_dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country],
    }

    data = pd.DataFrame.from_dict(data_dict)

    cat_features = categorical_features(data)

    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        keep_cat=keep_cat,
    )

    prediction = inference(classifier, X)
    prediction = labeler.inverse_transform(prediction)
    return prediction[0]
