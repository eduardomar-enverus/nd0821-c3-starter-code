import numpy as np
import sklearn

from project.ml.model import train_model, compute_model_metrics

X_array = np.array(
    [
        [3.80000e01, 5.96600e04, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [5.20000e01, 1.63027e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [2.70000e01, 1.26730e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [2.40000e01, 4.40750e04, 0.00000e00, 1.00000e00, 0.00000e00, 0.00000e00],
        [5.20000e01, 2.36180e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [3.20000e01, 1.13838e05, 0.00000e00, 0.00000e00, 1.00000e00, 0.00000e00],
        [3.40000e01, 3.81153e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [4.00000e01, 1.68936e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [2.20000e01, 3.08205e05, 0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
        [7.70000e01, 2.86780e04, 1.00000e00, 0.00000e00, 0.00000e00, 0.00000e00],
    ]
)
y_array = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1])

predictions = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])


def test_train_model():
    clf = train_model(X_array, y_array)
    assert type(clf) == sklearn.linear_model._logistic.LogisticRegression


def test_compute_model_metrics():
    precision, recall, fbeta = compute_model_metrics(y_array, predictions)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference():

    clf = sklearn.linear_model.LogisticRegression(random_state=42, max_iter=100).fit(X_array, y_array)
    predictions_model = clf.predict(X_array)
    expected_predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    np.array_equal(predictions_model, expected_predictions)
