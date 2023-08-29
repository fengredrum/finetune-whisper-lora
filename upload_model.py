from huggingface_hub import create_repo, HfApi

repo_id = "Oblivion208/whisper-tiny-cantonese"
create_repo(repo_id)

api = HfApi()
api.upload_folder(
    folder_path="./logs/whisper-tiny-cantonese",
    repo_id=repo_id,
)
