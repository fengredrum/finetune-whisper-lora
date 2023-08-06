from huggingface_hub import snapshot_download

repo_id = "openai/whisper-small"
root_dir = "./hf_hub/models/"

snapshot_download(repo_id=repo_id,
                  ignore_patterns=["*.msgpack", "*.h5", "*.pt"],
                  local_dir=root_dir + repo_id,
                  local_dir_use_symlinks=False)
