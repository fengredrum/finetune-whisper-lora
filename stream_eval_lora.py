import torch
import numpy as np
import evaluate
import gc

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Dataset, Audio
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm

# Model setups
peft_model_id = "./logs/whisper-large-lora-cantonese/checkpoint-1200"
task = "transcribe"
metric = evaluate.load("cer")
language = "zh"
# Dataset setups
dataset_name = "mdcc"
dataset_dir = "./datasets/" + dataset_name + "/"
num_samples = 1000
batch_size = 32

# TODO 8-bit training and inference very slow
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, task=task)
processor = WhisperProcessor.from_pretrained(
    peft_config.base_model_name_or_path, task=task)
feature_extractor = processor.feature_extractor
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)
# model.config.suppress_tokens = []


def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


filelist = dataset_dir + "cnt_asr_test_metadata.csv"
filepaths_and_text = load_filepaths_and_text(filelist)[:num_samples]
filepaths_and_text[0].append("transcription")
audio_paths, transcription_texts = [], []

for i in range(1, len(filepaths_and_text)):
    audio_path = dataset_dir + filepaths_and_text[i][0][2:]
    audio_paths.append(audio_path)

    transcription_path = dataset_dir + filepaths_and_text[i][1][2:]
    with open(transcription_path, encoding='utf-8') as f:
        transcription = [line.strip() for line in f][0]
    filepaths_and_text[i].append(transcription)
    transcription_texts.append(transcription)

dataset_dict = {"audio": audio_paths, "sentence": transcription_texts}
ds_test = Dataset.from_dict(dataset_dict)
ds_test.to_json(dataset_dir + "test.json")

ds_test = load_dataset("json", data_files=dataset_dir + "/test.json", split='train',
                       streaming=True, features=ds_test.features)

ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))
ds_test = ds_test.map(prepare_dataset, remove_columns=list(
    next(iter(ds_test)).keys())).with_format("torch")

seed, buffer_size = 42, 500
ds_test = ds_test.shuffle(seed, buffer_size=buffer_size)
print(next(iter(ds_test)))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(
    ds_test, batch_size=batch_size, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                ).cpu().numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    del generated_tokens, labels, batch
    gc.collect()
cer = 100 * metric.compute()
print(f"{cer=}")
