import torch
import evaluate

from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Seq2SeqTrainer
from peft import prepare_model_for_int8_training, LoraConfig, LoraConfig, get_peft_model
from datasets import load_dataset, IterableDatasetDict, Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union


# Model setups
model_id = "small"
experiment_name = f"whisper-{model_id}-lora-cantonese"
model_name_or_path = f"openai/whisper-{model_id}"
task = "transcribe"
metric = evaluate.load("cer")
language = "zh"
# Dataset setups
dataset_name = "mdcc"
dataset_dir = "./datasets/" + dataset_name + "/"
max_input_length = 30.0
num_test_samples = 1000

# Load pretrained
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task)


def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


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
ds_keys = ["train", "test"]
for key in ds_keys:
    filelist = dataset_dir + f"cnt_asr_{key}_metadata.csv"
    filepaths_and_text = load_filepaths_and_text(filelist)
    if key == "test":
        filepaths_and_text = filepaths_and_text[:num_test_samples]
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
    ds_tmp = Dataset.from_dict(dataset_dict)
    ds_tmp.to_json(dataset_dir + f"{key}.json")

    ds[key] = load_dataset("json", data_files=dataset_dir + f"/{key}.json", split='train',
                           streaming=True, features=ds_tmp.features)

ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.map(prepare_dataset, remove_columns=list(
    next(iter(ds.values())).features)).with_format("torch")


seed, buffer_size = 42, 500
ds = ds.shuffle(seed, buffer_size=buffer_size)

print(next(iter(ds["train"])))
print(next(iter(ds["test"])))

ds["train"] = ds["train"].filter(
    is_audio_in_length_range, input_columns=["input_length"])
ds["test"] = ds["test"].filter(
    is_audio_in_length_range, input_columns=["input_length"])


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
    per_device_train_batch_size=48,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=500,
    max_steps=10000,
    evaluation_strategy="steps",
    # gradient_checkpointing=True,
    # optim="adamw_torch",
    fp16=True,
    per_device_eval_batch_size=8,
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
