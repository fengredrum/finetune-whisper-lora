import torch
import evaluate

from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Seq2SeqTrainer
from peft import prepare_model_for_int8_training, LoraConfig, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from load_datasets import load_process_datasets


# Model setups
model_id = "large-v2"
experiment_name = f"whisper-{model_id}-lora-cantonese"
model_name_or_path = f"openai/whisper-{model_id}"
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
num_test_samples = 1000

# Load pretrained
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task)

ds = load_process_datasets(
    datasets_settings,
    processor,
    max_input_length=max_input_length,
    num_test_samples=num_test_samples,
)
print("train sample: ", next(iter(ds["train"])))
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
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=500,
    max_steps=10000,
    evaluation_strategy="steps",
    # gradient_checkpointing=True,
    # optim="adamw_torch",
    fp16=True,
    per_device_eval_batch_size=16,
    generation_max_length=225,
    save_steps=400,
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
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()
