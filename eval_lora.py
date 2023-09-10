import torch
import numpy as np
import evaluate
import gc

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from load_datasets import load_process_datasets

# Model setups
peft_model_id = "Oblivion208/whisper-large-v2-lora-mix"
task = "transcribe"
metric = evaluate.load("cer")
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
num_test_samples = 5000
batch_size = 64

peft_config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(
    peft_config.base_model_name_or_path, task=task, language=language)
processor = WhisperProcessor.from_pretrained(
    peft_config.base_model_name_or_path, task=task, language=language)
feature_extractor = processor.feature_extractor

ds = load_process_datasets(
    datasets_settings,
    processor,
    max_input_length=max_input_length,
    num_test_samples=num_test_samples,
    test_only=True,
    streaming=False,
    num_proc=4,
)
print("test sample: ", next(iter(ds["test"])))


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
    ds["test"], batch_size=batch_size, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)

# TODO 8-bit training and inference very slow
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)
# model.config.suppress_tokens = []
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
