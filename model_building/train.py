import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo

print("\n TRAINING PIPELINE STARTED\n")

# =====================================================
#  MLFLOW CONFIG
# =====================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "mlops-training-experiment")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# =====================================================
#  HF AUTH
# =====================================================
HF_TOKEN = os.getenv("MLOps")
api = HfApi(token=HF_TOKEN)

# =====================================================
#  LOAD DATA
# =====================================================
DATA_REPO_ID = "tushar77more/tourism_project_dataset"

Xtrain = load_dataset(DATA_REPO_ID, data_files="Xtrain.csv", split="train").to_pandas()
Xtest  = load_dataset(DATA_REPO_ID, data_files="Xtest.csv", split="train").to_pandas()
ytrain = load_dataset(DATA_REPO_ID, data_files="ytrain.csv", split="train").to_pandas().values.ravel()
ytest  = load_dataset(DATA_REPO_ID, data_files="ytest.csv", split="train").to_pandas().values.ravel()

# CLEANUP: Drop Unnamed columns if they snuck in
Xtrain = Xtrain.loc[:, ~Xtrain.columns.str.contains('^Unnamed')]
Xtest = Xtest.loc[:, ~Xtest.columns.str.contains('^Unnamed')]

# =====================================================
#  PREPROCESSING
# =====================================================
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
    'MaritalStatus', 'Designation'
]

# Define numerical columns EXACTLY as they appear in app.py
numerical_cols = [
    'Age', 'MonthlyIncome', 'NumberOfTrips', 'CityTier', 
    'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
    'PreferredPropertyStar', 'NumberOfChildrenVisiting', 'OwnCar', 
    'Passport', 'PitchSatisfactionScore'
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# =====================================================
#  MODEL PIPELINE
# =====================================================
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

param_grid = {
    "classifier__n_estimators": [100],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.1],
}

# =====================================================
#  TRAIN AND TRACK
# =====================================================
if mlflow.active_run():
    mlflow.end_run()
    
with mlflow.start_run(run_name="Tourism_XGB_Run"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    y_pred_test = best_model.predict(Xtest)

    metrics = {
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_f1": f1_score(ytest, y_pred_test),
        "test_auc": roc_auc_score(ytest, best_model.predict_proba(Xtest)[:, 1]),
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(best_model, artifact_path="model")

# =====================================================
#  SAVE FOR GITHUB ACTIONS (LOCAL ROOT)
# =====================================================
# We save to the root directory so the YAML 'path: tourism_xgb_model.pkl' works!
model_filename = "tourism_xgb_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Local model saved to: {os.path.abspath(model_filename)}")

# =====================================================
#  HF MODEL REPO UPLOAD (OPTIONAL BACKUP)
# =====================================================
try:
    MODEL_REPO_ID = "tushar77more/tourism_model"
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print("Model uploaded to HF Model Registry")
except Exception as e:
    print(f" HF Model Registry upload skipped/failed: {e}")

print("\n TRAINING COMPLETE")
