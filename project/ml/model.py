from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from contextlib import redirect_stdout


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = LogisticRegression(random_state=42, max_iter=5000).fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slice_inference(predictions, y, data, categorical_features, printed=False):
    if printed:
        with open("slice_output.txt", "w") as f:
            for category in categorical_features:
                for cat_value in sorted(data[category].unique()):
                    mask = data[data[category] == cat_value].index
                    mask_predictions = predictions[mask]
                    mask_y = y[mask]

                    precision, recall, fbeta = compute_model_metrics(
                        mask_y, mask_predictions
                    )

                    with redirect_stdout(f):
                        print(
                            f"Slice metrics - Feature: {category} - Value: {cat_value}"
                        )
                        print(f"Category percent {round(100*len(mask)/len(data),2)}")
                        print(f"Precision = {precision}")
                        print(f"Recall = {recall}")
                        print(f"FBeta = {fbeta}")
                        print()
