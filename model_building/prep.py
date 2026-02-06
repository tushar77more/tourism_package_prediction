# for data manipulation
import pandas as pd
from datasets import load_dataset
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("MLOps"))
#DATASET_PATH = "hf://datasets/tushar77more/tourism_project_dataset/tourism.csv"

#df = pd.read_csv(DATASET_PATH)

dataset = load_dataset(
    "tushar77more/tourism_project_dataset",
    data_files="tourism.csv",
    split="train"
)

df = dataset.to_pandas()

print("Dataset loaded successfully.")
print("Initial Shape:", df.shape)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop identifier column
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Drop fully empty columns (safety)
df.dropna(axis=1, how='all', inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Shape after cleaning:", df.shape)

# ---------------------------------
# Encode Categorical Columns
# ---------------------------------
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
    print(f" Encoded: {col}")

# ---------------------------------
# Define Target Variable
# ---------------------------------
target_col = "ProdTaken"  # Whether customer purchased package

if target_col not in df.columns:
    raise ValueError(" Target column not found!")

X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------------
# Train-Test Split
# ---------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(" Train/Test split done")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)

# ---------------------------------
# Save Locally
# ---------------------------------
os.makedirs("/content/drive/MyDrive/tourism_project/data/processed", exist_ok=True)

Xtrain_path = "/content/drive/MyDrive/tourism_project/data/processed/Xtrain.csv"
Xtest_path = "/content/drive/MyDrive/tourism_project/data/processed/Xtest.csv"
ytrain_path = "/content/drive/MyDrive/tourism_project/data/processed/ytrain.csv"
ytest_path = "/content/drive/MyDrive/tourism_project/data/processed/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print(" Files saved locally")

# ---------------------------------
# Upload to Hugging Face
# ---------------------------------
files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="tushar77more/tourism_project_dataset",
        repo_type="dataset",
    )
    print(f"☁ Uploaded: {file_path}")

print("🎉 Preprocessing pipeline completed successfully!")
