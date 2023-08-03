import torch
import numpy as np
import evaluate
import gc

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import load_dataset, DatasetDict, Audio
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm

metric = evaluate.load("cer")
peft_model_id = "./logs/whisper-small-cantonese/checkpoint-1000"
language = "zh"
language_abbr = "zh-HK"
task = "transcribe"
dataset_name = "common_voice"
batch_size = 32

# TODO 8-bit training and inference very slow
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, task=task)
processor = WhisperProcessor.from_pretrained(
    peft_config.base_model_name_or_path, task=task)
feature_extractor = processor.feature_extractor
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)
# model.config.suppress_tokens = []

common_voice = DatasetDict()
common_voice["test"] = load_dataset(
    dataset_name, language_abbr, split="test[:100]")
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes",
        "gender", "locale", "path", "segment", "up_votes"]
)
print(common_voice)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=2)


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
    common_voice["test"], batch_size=batch_size, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
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
