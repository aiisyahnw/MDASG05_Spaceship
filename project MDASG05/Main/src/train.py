"""
ASG 04 – Step 3: Training
Trains a Logistic Regression classifier and logs to MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def train(train_data, preprocessor):

    X_train, y_train = train_data

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/model.pkl")

    print("Training selesai.")

    return pipeline