# Script to train machine learning model.
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from project3.ml.data import process_data, categorical_features

# Add the necessary imports for the delete code.
from project3.ml.model import train_model, compute_model_metrics, slice_inference

data = pd.read_csv("../data/census_clean.csv", index_col=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.

train, test = train_test_split(data, test_size=0.20)

cat_features = categorical_features(data)

# Prrocess training data
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Process the test data with the process_data function.
X_test, y_test, encoder, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
)

# Train model.
clf = train_model(X_train, y_train)

# Save model
with open("../model/model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Predictions
predictions = clf.predict(X_test)

# General Metrics
y_test = lb.fit_transform(y_test.values).ravel()
precision, recall, fbeta = compute_model_metrics(y_test, predictions)

print("General metrics")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"FBeta = {fbeta}")

# Slice Metrics
slice_inference(predictions, y_test, test.reset_index(drop=True), cat_features)