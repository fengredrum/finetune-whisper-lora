import torch
import evaluate

from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union


# Model setups
experiment_name = "whisper-small-cantonese"
model_name_or_path = "openai/whisper-small"
task = "transcribe"
metric = evaluate.load("cer")
language = "zh"
# Dataset setups
dataset_name = "mozilla-foundation/common_voice_11_0"
language_abbr = "zh-HK"
saved_dir = "./hf_hub/datasets/" + dataset_name + "/" + language_abbr
num_test_samples = 1000

# Load test dataset
try:
    ds = DatasetDict()
    ds["test"] = load_from_disk(saved_dir)["test"]
except:
    print("Download dataset...")
    ds = DatasetDict()
    ds["train"] = load_dataset(
        dataset_name, language_abbr, split="train+validation")
    ds["test"] = load_dataset(dataset_name, language_abbr, split="test")

ds = ds.remove_columns(
    ["accent", "age", "client_id", "down_votes",
     "gender", "locale", "path", "segment", "up_votes"]
)
ds["test"] = Dataset.from_dict(ds["test"][:num_test_samples])
print(ds)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


ds = ds.map(
    prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)


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


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/" + experiment_name,  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=5,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    optim="adamw_torch",
    fp16=True,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()
