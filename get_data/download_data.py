from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
import os


repo_id = "Manoli201/EPFHWD"  # <-- change this
repo_type = "dataset"
local_data_dir = os.path.join(os.path.dirname(__file__), "..", "data/")

all_files = list_repo_files(repo_id="Manoli201/EPFHWD", repo_type="dataset")
image_files = [f for f in all_files if f.startswith("images/")]

os.makedirs(local_data_dir, exist_ok=True)

for filename in image_files:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=local_data_dir,
        local_dir_use_symlinks=False  # set to False to copy files instead of symlinking
    )
    print(f"Downloaded {filename} to {local_path}")