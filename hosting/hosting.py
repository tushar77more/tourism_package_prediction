from huggingface_hub import HfApi
import os

folder_path = "deployment"

api = HfApi(token=os.getenv("MLOps"))
api.upload_folder(
    folder_path=folder_path,     # the local folder containing your files
    repo_id="tushar77more/tourism-package-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
