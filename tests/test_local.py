import json
import os
import pytest

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture(scope='session')
def greater_than_fifty():
    fake_db1 = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States",
        "salary": ">50K",
    }
    return fake_db1

fake_db2 = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 83311,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 13,
    "native-country": "United-States",
    "salary": "<=50K",
}


def test_get_path():
    r = client.get("/")
    print(os.getcwd())
    assert r.status_code == 200
    assert r.json() == "Hello World!"


def test_pred(greater_than_fifty):
    r = client.post("/predict_salary/", json=greater_than_fifty)
    # assert r.status_code == 200
    assert r.json() == ">50K"


def test_pred2():
    r = client.post("/predict_salary/", json=fake_db2)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50k"}
