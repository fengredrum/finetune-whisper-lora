import torch
import numpy as np
import evaluate
import argparse
import gc

from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from load_datasets import load_process_datasets


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


# TODO Move to ArgumentParser
datasets_settings = [
    ["mdcc", {}],
    ["common_voice", {"language_abbr": "zh-HK"}],
    ["aishell_1", {}],
    ["thchs_30", {}],
    ["magicdata", {}],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--model_name_or_path",
                        default="Oblivion208/whisper-tiny-cantonese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_new_tokens", default=255, type=int)
    parser.add_argument("--metric", default="cer")
    parser.add_argument("--device", default="cuda")
    # Dataset setups
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=30.0, type=float)
    parser.add_argument("--test_only", default=True, type=bool)
    parser.add_argument("--streaming", default=False, type=bool)
    parser.add_argument("--num_proc", default=4, type=int)

    args = parser.parse_args()
    print(f"Settings: {args}")

    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name_or_path, task=args.task, language=args.language)
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, task=args.task, language=args.language)
    feature_extractor = processor.feature_extractor

    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=args.max_input_length,
        num_test_samples=args.num_test_samples,
        test_only=args.test_only,
        streaming=args.streaming,
        num_proc=args.num_proc,
    )
    print("test sample: ", next(iter(ds["test"])))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        ds["test"], batch_size=args.batch_size, collate_fn=data_collator)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path).to(args.device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # model.config.use_cache = False
    model.eval()

    metric = evaluate.load(args.metric)
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to(args.device),
                    return_dict_in_generate=True,
                    max_new_tokens=args.max_new_tokens,
                )
            ).sequences.cpu().numpy()

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
