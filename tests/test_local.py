import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(scope='session')
def greater_than_fifty():
    sample = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return sample


@pytest.fixture(scope='session')
def less_than_fifty():
    sample = {
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
        "native_country": "United-States"
    }
    return sample



def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello World!"


def test_pred(greater_than_fifty):
    r = client.post("/predict_salary/", json=greater_than_fifty)
    assert r.status_code == 200
    assert r.json() == ">50K"


def test_pred2(less_than_fifty):
    r = client.post("/predict_salary/", json=less_than_fifty)
    assert r.status_code == 200
    assert r.json() == "<=50K"
