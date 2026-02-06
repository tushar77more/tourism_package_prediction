import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("MLOps"))

folder_path = "/content/drive/MyDrive/tourism_project/data"

# Check local files
print("Local files:", os.listdir(folder_path))

repo_id = "tushar77more/tourism_project_dataset"

# Optional: see files already on HF
print("Files in HF repo:", api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Initial dataset upload"
)

print("✅ Dataset files uploaded")
