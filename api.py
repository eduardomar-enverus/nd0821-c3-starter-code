import json
import requests

URL = "https://udacity-emr.herokuapp.com/predict_salary/"

sample_prediction = {
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
}


response = requests.post(URL, data=json.dumps(sample_prediction))

dictionary = {
    "REQUEST BODY": json.dumps(sample_prediction),
    "STATUS CODE": response.status_code,
    "PREDICTION": response.json()
}

print(json.dumps(dictionary, indent=4))
