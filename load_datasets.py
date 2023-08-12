from datasets import load_dataset, concatenate_datasets, IterableDatasetDict, Dataset, Audio
from tqdm import tqdm


def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_mdcc(dataset_root, sampling_rate=16000, use_valid_to_train=True):
    ds = IterableDatasetDict()

    dataset_dir = dataset_root + "mdcc/"
    ds_keys = ["train", "valid", "test"]
    for key in ds_keys:
        filelist = dataset_dir + f"cnt_asr_{key}_metadata.csv"
        filepaths_and_text = load_filepaths_and_text(filelist)[:100]
        filepaths_and_text[0].append("transcription")
        audio_paths, transcription_texts = [], []

        for i in tqdm(range(1, len(filepaths_and_text))):
            audio_path = dataset_dir + filepaths_and_text[i][0][2:]
            audio_paths.append(audio_path)

            transcription_path = dataset_dir + filepaths_and_text[i][1][2:]
            with open(transcription_path, encoding='utf-8') as f:
                transcription = [line.strip() for line in f][0]
            filepaths_and_text[i].append(transcription)
            transcription_texts.append(transcription)

        dataset_dict = {"audio": audio_paths, "sentence": transcription_texts}
        ds_tmp = Dataset.from_dict(dataset_dict)
        ds_tmp.to_json(dataset_dir + f"{key}.json")

        ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                               streaming=True, features=ds_tmp.features)

    if use_valid_to_train:
        ds["train"] = concatenate_datasets([ds["train"], ds["valid"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def load_common_voice(language_abbr="zh-HK", sampling_rate=16000, use_valid_to_train=True):
    dataset_name = "mozilla-foundation/common_voice_11_0"
    ds = IterableDatasetDict()

    ds["train"] = load_dataset(
        dataset_name, language_abbr, split="train", streaming=True, use_auth_token=True)
    ds["test"] = load_dataset(
        dataset_name, language_abbr, split="test", streaming=True, use_auth_token=True)
    if use_valid_to_train:
        ds["valid"] = load_dataset(
            dataset_name, language_abbr, split="validation", streaming=True)
        ds["train"] = concatenate_datasets([ds["train"], ds["valid"]])

    ds = ds.remove_columns(
        ["accent", "age", "client_id", "down_votes",
         "gender", "locale", "path", "segment", "up_votes"]
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return ds


def load_process_datasets(datasets_name, processor, dataset_root="./datasets/", num_test_samples=1000,
                          sampling_rate=16000, max_input_length=30.0, buffer_size=500, seed=42):

    def prepare_dataset(batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # optional pre-processing steps
        transcription = batch["sentence"]

        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    def is_audio_in_length_range(length):
        return length < max_input_length

    ds = IterableDatasetDict()
    train_list, test_list = [], []
    for name in datasets_name:
        ds_tmp = None
        if name == "mdcc":
            ds_tmp = load_mdcc(dataset_root, sampling_rate=sampling_rate)
            print("mdcc: ", next(iter(ds_tmp["train"])))
        elif name == "common_voice":
            ds_tmp = load_common_voice(
                language_abbr="zh-HK", sampling_rate=sampling_rate)
            print("common_voice: ", next(iter(ds_tmp["train"])))

        if ds_tmp is not None:
            train_list.append(ds_tmp["train"])
            test_list.append(ds_tmp["test"])

    ds["train"] = concatenate_datasets(train_list)
    ds["test"] = concatenate_datasets(test_list)
    ds = ds.map(prepare_dataset, remove_columns=list(
        next(iter(ds.values())).features)).with_format("torch")

    ds = ds.filter(
        is_audio_in_length_range, input_columns=["input_length"])
    ds["test"] = ds["test"].take(num_test_samples)
    ds = ds.shuffle(seed, buffer_size=buffer_size)
    return ds
