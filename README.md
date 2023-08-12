# Finetune-Whisper-Lora

Download MDCC dataset

```bash
mkdir -p datasets/mdcc
cd datasets/mdcc
wget https://storage.googleapis.com/samcah-bucket/cantonese-asr/cantonese_dataset.zip
unzip cantonese_dataset.zip
rm cantonese_dataset.zip
```

> Cantonse-ASR: Yu, Tiezheng, Frieske, Rita, Xu, Peng, Cahyawijaya, Samuel, Yiu, Cheuk Tung, Lovenia, Holy, Dai, Wenliang, Barezi, Elham, Chen, Qifeng, Ma, Xiaojuan, Shi, Bertram, Fung, Pascale (2022) "Automatic Speech Recognition Datasets in Cantonese: A Survey and New Dataset", 2022. Link: https://arxiv.org/pdf/2201.02419.pdf

## Approximate Performance Evaluation

### Cantonese Test Results Comparison

#### MDCC

| Model name                      | Parameters | Finetune Steps | Time Spend | Training Loss | Validation Loss | CER % |
| ------------------------------- | ---------- | -------------- | ---------- | ------------- | --------------- | ----- |
| whisper-tiny-cantonese          | 39 M       | 4400           | 6h 29m     | 0.0135        | 0.209           | 9.72  |
| whisper-base-cantonese          | 74 M       | 2400           | 3h 12m     | 0.0407        | 0.156           | 8.48  |
| whisper-small-cantonese         | 244 M      | 3600           | 6h 38m     | 0.0266        | 0.137           | 6.16  |
| whisper-large-v2-lora-cantonese | 15 M       | 2000           | 11h 32m    | 0.0408        | 0.093           | 4.58  |

#### Common Voice Corpus 11.0

| Model name                      | Original CER % | w/o Finetune CER % | Jointly Finetune CER % |
| ------------------------------- | -------------- | ------------------ | ---------------------- |
| whisper-tiny-cantonese          | 124.03         | 66.85              |                        |
| whisper-base-cantonese          | 78.24          | 61.42              |                        |
| whisper-small-cantonese         | 52.83          | 31.23              |                        |
| whisper-large-v2-lora-cantonese | 37.53          | 19.38              |                        |
