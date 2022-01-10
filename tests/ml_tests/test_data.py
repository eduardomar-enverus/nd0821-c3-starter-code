import pandas as pd
import numpy as np
import sklearn

from project.ml.data import categorical_features, process_data


def test_categorical_features():
    df = pd.DataFrame(
        {
            "Cat_1": ["a", "b", "c"],
            "Cat_2": ["a", "b", "c"],
            "Num_1": [1, 2, 3],
            "Num_2": [1.0, 2.0, 3.0],
        }
    )
    cat_list = categorical_features(df)
    expected_list = ["Cat_1", "Cat_2"]

    assert cat_list == expected_list


def test_process_data():

    df = pd.DataFrame(
        {
            "age": [38, 52, 27, 24, 52, 32, 34, 40, 22, 77],
            "workclass": [
                "Private",
                "Private",
                "Private",
                "Federal-gov",
                "Private",
                "Local-gov",
                "Private",
                "Private",
                "Private",
                "?",
            ],
            "fnlgt": [
                59660,
                163027,
                126730,
                44075,
                236180,
                113838,
                381153,
                168936,
                308205,
                28678,
            ],
            "salary": [
                ">50K",
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                "<=50K",
                "<=50K",
                ">50K",
            ],
        }
    )
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
    X, y, encoder, lb = process_data(df, ["workclass"], ["workclass"], label="salary")

    np.testing.assert_array_equal(X, X_array)
    np.testing.assert_array_equal(y, y_array)
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
