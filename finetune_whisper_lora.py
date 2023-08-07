import torch
import evaluate

from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Seq2SeqTrainer
from peft import prepare_model_for_int8_training, LoraConfig, LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

metric = evaluate.load("cer")
experiment_name = "whisper-large-lora-cantonese"
model_name_or_path = "openai/whisper-large-v2"
language = "zh"
language_abbr = "zh-HK"
task = "transcribe"
dataset_name = "common_voice"

common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    dataset_name, language_abbr, split="train+validation")
common_voice["test"] = load_dataset(
    dataset_name, language_abbr, split="test[:1000]")

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes",
        "gender", "locale", "path", "segment", "up_votes"]
)
print(common_voice)

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task)

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
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)


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


# TODO 8-bit training and inference very slow
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)

config = LoraConfig(r=32, lora_alpha=64, target_modules=[
                    "q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/" + experiment_name,  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=500,
    num_train_epochs=5,
    evaluation_strategy="steps",
    # gradient_checkpointing=True,
    # optim="adamw_torch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    remove_unused_columns=False,
    label_names=["labels"],  # same reason as above
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()
