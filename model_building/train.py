
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
#  MLFLOW CONFIG (DEV + PROD SAFE)
# =====================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "mlops-training-experiment")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print(f" MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f" Experiment: {MLFLOW_EXPERIMENT_NAME}")

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

mlflow.log_param("dataset_repo", DATA_REPO_ID)

# =====================================================
#  PREPROCESSING
# =====================================================
categorical_cols = Xtrain.select_dtypes(include="object").columns.tolist()
numerical_cols = Xtrain.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# =====================================================
#  MODEL
# =====================================================
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.01, 0.1],
    "classifier__subsample": [0.8, 1],
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

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="Tourism_Purchase_Predictor"
    )

    print("\n Metrics:\n", json.dumps({k: float(v) for k,v in metrics.items()}, indent=4))

# =====================================================
#  SAVE AND HF UPLOAD
# =====================================================
os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/tourism_xgb_model.pkl"
joblib.dump(best_model, model_path)

MODEL_REPO_ID = "tushar77more/tourism_model"
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_xgb_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)

print("\n TRAINING , TRACKING AND REGISTRATION COMPLETE")
