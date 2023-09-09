# Finetune-Whisper-LoRA

<p align="left">
🤗 <a href="https://huggingface.co/Oblivion208" target="_blank">HF Repo</a>  •🐱 <a href="https://github.com/fengredrum/finetune-whisper-lora" target="_blank">Github Repo</a> 
</p>

## Get Started

### 1. Setup Docker Environment

Switch to the docker folder and build Docker GPU image for training:

```bash
cd docker
docker compose build
```

Onece the building process complete, run the following command to start a Docker container and attach to it:

```bash
docker compose up -d
docker exec -it asr bash
```

### 2. Prepare Training Data

See detail in [dataset_scripts](dataset_scripts\README.md) folder.

### 3. Finetune Pretrained Model

```bash
python finetune.py # Finetuning
```

```bash
python finetune_lora.py # LoRA Finetuning
```

### 4. Evaluate Performance

```bash
python eval.py # Evaluation
```

```bash
python eval_lora.py # LoRA Evaluation
```

## Approximate Performance Evaluation

The following models are all trained and evaluated on a single RTX 3090 GPU via [Vast.ai](https://cloud.vast.ai/?ref_id=78038 "Vast.ai").

### Cantonese Test Results Comparison

#### MDCC

| Model name                      | Parameters | Finetune Steps | Time Spend | Training Loss | Validation Loss | CER % | Finetuned Model                                                                                                          |
| ------------------------------- | ---------- | -------------- | ---------- | ------------- | --------------- | ----- | ------------------------------------------------------------------------------------------------------------------------ |
| whisper-tiny-cantonese          | 39 M       | 3200           | 4h 34m     | 0.0485        | 0.771           | 11.10 | [Link](https://huggingface.co/Oblivion208/whisper-tiny-cantonese "Oblivion208/whisper-tiny-cantonese")                   |
| whisper-base-cantonese          | 74 M       | 7200           | 13h 32m    | 0.0186        | 0.477           | 7.66  | [Link](https://huggingface.co/Oblivion208/whisper-base-cantonese "Oblivion208/whisper-base-cantonese")                   |
| whisper-small-cantonese         | 244 M      | 3600           | 6h 38m     | 0.0266        | 0.137           | 6.16  | [Link](https://huggingface.co/Oblivion208/whisper-small-cantonese "Oblivion208/whisper-small-cantonese")                 |
| whisper-small-lora-cantonese    | 3.5 M      | 8000           | 21h 27m    | 0.0687        | 0.382           | 7.40  | [Link](https://huggingface.co/Oblivion208/whisper-small-lora-cantonese "Oblivion208/whisper-small-lora-cantonese")       |
| whisper-large-v2-lora-cantonese | 15 M       | 10000          | 33h 40m    | 0.0046        | 0.277           | 3.77  | [Link](https://huggingface.co/Oblivion208/whisper-large-v2-lora-cantonese "Oblivion208/whisper-large-v2-lora-cantonese") |

#### Common Voice Corpus 11.0

| Model name                      | Original CER % | w/o Finetune CER % | Jointly Finetune CER % |
| ------------------------------- | -------------- | ------------------ | ---------------------- |
| whisper-tiny-cantonese          | 124.03         | 66.85              | 35.87                  |
| whisper-base-cantonese          | 78.24          | 61.42              | 16.73                  |
| whisper-small-cantonese         | 52.83          | 31.23              | /                      |
| whisper-small-lora-cantonese    | 37.53          | 19.38              | 14.73                  |
| whisper-large-v2-lora-cantonese | 37.53          | 19.38              | 9.63                   |

## Requirements

- Transformers
- Accelerate
- Datasets
- PEFT
- bitsandbytes
- librosa

## References

1. https://github.com/openai/whisper
2. https://huggingface.co/blog/fine-tune-whisper
3. https://huggingface.co/docs/peft/task_guides/int8-asr
4. https://huggingface.co/alvanlii/whisper-largev2-cantonese-peft-lora
