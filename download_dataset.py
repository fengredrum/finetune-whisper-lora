from datasets import load_dataset, DatasetDict, load_from_disk

dataset_name = "mozilla-foundation/common_voice_11_0"
language_abbr = "zh-HK"
saved_dir = "./hf_hub/datasets/" + dataset_name + "/" + language_abbr

ds = DatasetDict()
ds["train"] = load_dataset(
    dataset_name, language_abbr, split="train+validation")
ds["test"] = load_dataset(
    dataset_name, language_abbr, split="test")
print(ds)
ds.save_to_disk(saved_dir, num_shards={'train': 16, 'test': 4}, num_proc=2)

local_ds = load_from_disk(saved_dir)
print(local_ds)
