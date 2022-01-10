from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


fake_db1 = {
    "age": 25,
    "fnlgt": 30,
    "education_num": 12,
    "capital_gain": 20000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "x3_Female": "0",
    "x3_Male": "1",
}

fake_db2 = {
    "age": 40,
    "fnlgt": 2.7224e05,
    "education_num": 9,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "x3_Female": "1",
    "x3_Male": "0",
}


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_pred():
    r = client.post("/predict/", json=fake_db1)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50k"}


def test_pred2():
    r = client.post("/predict/", json=fake_db2)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50k"}
