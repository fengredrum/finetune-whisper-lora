import os

from datasets import (load_dataset, concatenate_datasets,
                      IterableDatasetDict, DatasetDict, Dataset, Audio)
from tqdm import tqdm


def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_magicdata(dataset_root, sampling_rate=16000, streaming=True, use_valid_to_train=False, test_only=False):
    ds = IterableDatasetDict()

    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]

    audio_paths, transcription_texts = {}, {}
    for key in ds_keys:
        dataset_dir = dataset_root + "magicdata/" + key + "/"
        if os.path.exists(dataset_dir):
            filepaths_and_text = load_filepaths_and_text(
                dataset_dir + "TRANS.txt", split="\t")
            audio_paths[key], transcription_texts[key] = [], []
            for filename, subdir, text in filepaths_and_text[1:]:
                audio_path = dataset_dir + subdir + "/" + filename
                if os.path.exists(audio_path):
                    audio_paths[key].append(audio_path)
                    transcription_texts[key].append(text)
                else:
                    print(
                        f"Skip file: {audio_path}, file path does not exist.")

            dataset_dict = {
                "audio": audio_paths[key], "sentence": transcription_texts[key]}
            ds_tmp = Dataset.from_dict(dataset_dict)
            ds_tmp.to_json(dataset_dir + f"{key}.json", index=False)

            ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                                   streaming=streaming, features=ds_tmp.features)

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def load_thchs_30(dataset_root, sampling_rate=16000, streaming=True, use_valid_to_train=False, test_only=False):
    ds = IterableDatasetDict()

    dataset_dir = dataset_root + "thchs_30/data_thchs30/"
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]

    def load_transcripts(filename):
        with open(filename, encoding='utf-8') as f:
            texts = [line.strip() for line in f]
        return texts[0].replace(" ", "")

    audio_paths, transcription_texts = {}, {}
    list_dirs = os.walk(dataset_dir)
    for root, dirs, files in list_dirs:
        subset_name = root.split("/")[-1]
        if subset_name in ds_keys:
            audio_paths[subset_name] = [dataset_dir + subset_name + "/" +
                                        file for file in files if "wav" in file and "trn" not in file]
            transcription_texts[subset_name] = [load_transcripts(audio_path.replace(
                subset_name, "data") + ".trn") for audio_path in audio_paths[subset_name]]

    for key in ds_keys:
        dataset_dict = {
            "audio": audio_paths[key], "sentence": transcription_texts[key]}
        ds_tmp = Dataset.from_dict(dataset_dict)
        ds_tmp.to_json(dataset_dir + f"{key}.json", index=False)

        ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                               streaming=streaming, features=ds_tmp.features)

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    return ds


def load_aishell_1(dataset_root, sampling_rate=16000, streaming=True, use_valid_to_train=False, test_only=False):
    ds = IterableDatasetDict()

    dataset_dir = dataset_root + "aishell_1/data_aishell/"
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]

    def load_transcripts(filename, split=" ", maxsplit=1):
        with open(filename, encoding='utf-8') as f:
            filename_and_texts = [line.strip().split(
                split, maxsplit=maxsplit) for line in f]
        return filename_and_texts

    filelist = dataset_dir + "transcript/aishell_transcript_v0.8.txt"
    filename_and_texts = load_transcripts(filelist)
    dirpaths = []
    list_dirs = os.walk(dataset_dir)
    for root, dirs, files in list_dirs:
        if "S" in root:
            dirpaths.append(root)

    sid_dict = {}
    for i in range(len(dirpaths)):
        split_path = dirpaths[i].split("/")
        subset_name = split_path[-2]
        sid = split_path[-1]
        sid_dict[sid] = subset_name

    audio_paths, transcription_texts = {}, {}
    for key in ds_keys:
        audio_paths[key] = []
        transcription_texts[key] = []
    for filename, text in filename_and_texts:
        sid = "S" + filename.split("W")[0].split("S")[-1]
        subset_name = sid_dict[sid]
        audio_path = dataset_dir + "wav/" + subset_name + \
            "/" + sid + "/" + filename + ".wav"
        if subset_name in ds_keys:
            audio_paths[subset_name].append(audio_path)
            transcription_texts[subset_name].append(text.replace(" ", ""))

    for key in ds_keys:
        dataset_dict = {
            "audio": audio_paths[key], "sentence": transcription_texts[key]}
        ds_tmp = Dataset.from_dict(dataset_dict)
        ds_tmp.to_json(dataset_dir + f"{key}.json", index=False)

        ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                               streaming=streaming, features=ds_tmp.features)

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def load_mdcc(dataset_root, sampling_rate=16000, streaming=True, use_valid_to_train=True, test_only=False):
    ds = IterableDatasetDict()

    dataset_dir = dataset_root + "mdcc/"
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "valid", "test"]
    for key in ds_keys:
        filelist = dataset_dir + f"cnt_asr_{key}_metadata.csv"
        filepaths_and_text = load_filepaths_and_text(filelist)
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
        ds_tmp.to_json(dataset_dir + f"{key}.json", index=False)

        ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                               streaming=streaming, features=ds_tmp.features)

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["valid"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def load_common_voice(language_abbr="zh-HK", sampling_rate=16000, streaming=True, use_valid_to_train=True, test_only=False):
    dataset_name = "mozilla-foundation/common_voice_11_0"
    ds = IterableDatasetDict()

    ds["train"] = load_dataset(
        dataset_name, language_abbr, split="train", streaming=streaming, use_auth_token=True)
    ds["test"] = load_dataset(
        dataset_name, language_abbr, split="test", streaming=streaming, use_auth_token=True)
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


def load_process_datasets(datasets_settings, processor, dataset_root="./datasets/", streaming=True, test_only=False, num_test_samples=1000,
                          sampling_rate=16000, max_input_length=30.0, num_proc=2, buffer_size=500, seed=42):

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

    if streaming:
        ds = IterableDatasetDict()
    else:
        ds = DatasetDict()

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(name, kwargs)
        ds_tmp = None
        if name == "mdcc":
            ds_tmp = load_mdcc(
                dataset_root, sampling_rate=sampling_rate, test_only=test_only)
            print("mdcc: ", next(iter(ds_tmp["test"])))
        elif name == "common_voice":
            ds_tmp = load_common_voice(
                sampling_rate=sampling_rate, test_only=test_only, **kwargs)
            print(f"common_voice-{kwargs}: ", next(iter(ds_tmp["test"])))
        elif name == "aishell_1":
            ds_tmp = load_aishell_1(
                dataset_root, sampling_rate=sampling_rate, test_only=test_only)
            print("aishell_1: ", next(iter(ds_tmp["test"])))
        elif name == "magicdata":
            ds_tmp = load_magicdata(
                dataset_root, sampling_rate=sampling_rate, test_only=test_only)
            print("magicdata: ", next(iter(ds_tmp["test"])))
        elif name == "thchs_30":
            ds_tmp = load_thchs_30(
                dataset_root, sampling_rate=sampling_rate, streaming=streaming, test_only=test_only)
            print("thchs_30: ", next(iter(ds_tmp["test"])))

        if ds_tmp is not None:
            test_list.append(ds_tmp["test"])
            if not test_only:
                train_list.append(ds_tmp["train"])

    ds["test"] = concatenate_datasets(test_list)
    if not test_only:
        ds["train"] = concatenate_datasets(train_list)

    if streaming:
        ds = ds.map(prepare_dataset,
                    remove_columns=list(next(iter(ds.values())).features),
                    ).with_format("torch")
        ds = ds.filter(
            is_audio_in_length_range, input_columns=["input_length"])
        ds = ds.shuffle(seed, buffer_size=buffer_size)
        ds["test"] = ds["test"].take(num_test_samples)
    else:
        ds["test"] = ds["test"].select(range(num_test_samples))
        ds = ds.map(prepare_dataset,
                    remove_columns=ds["test"].column_names,
                    num_proc=num_proc,
                    ).with_format("torch")

    return ds


if __name__ == "__main__":
    from transformers import WhisperProcessor, WhisperTokenizer

    # Model setups
    model_name_or_path = "Oblivion208/whisper-tiny-cantonese"
    task = "transcribe"
    language = "zh"
    # Dataset setups
    datasets_settings = [
        ["mdcc", {}],
        ["common_voice", {"language_abbr": "zh-HK"}],
        ["aishell_1", {}],
        ["thchs_30", {}],
        ["magicdata", {}],
    ]
    max_input_length = 30.0
    num_test_samples = 100
    batch_size = 64

    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task)
    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=max_input_length,
        num_test_samples=num_test_samples,
        test_only=True,
        streaming=False,
        num_proc=2,
    )
    print(ds)
