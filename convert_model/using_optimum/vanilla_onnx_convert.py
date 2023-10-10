import argparse

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig


def save_pretrained(args):
    if args.peft_model_id:
        peft_config = PeftConfig.from_pretrained(args.peft_model_id)

        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                                     task=args.task, language=args.language)
        processor.save_pretrained(args.output_path)

        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map=args.device,
        )
        model = PeftModel.from_pretrained(model, args.peft_model_id)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            task=args.task, language=args.language)

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map=args.device,
        )
        model.save_pretrained(args.output_path)


def export_model(args):
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        args.output_path, export=True)
    model.save_pretrained(args.output_path)
    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id",
                        default="Oblivion208/whisper-base-cantonese", type=str)
    parser.add_argument("--peft_model_id", type=str)
    parser.add_argument("--task", default="transcribe", type=str)
    parser.add_argument("--language", default="zh", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--output_path",
                        default="models/vanilla_onnx", type=str)

    args = parser.parse_args()
    print(f"Settings: {args}")

    save_pretrained(args)
    export_model(args)
