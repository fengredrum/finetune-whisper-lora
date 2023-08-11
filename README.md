# Finetune-Whisper-Lora

Download MDCC dataset

```bash
wget https://storage.googleapis.com/samcah-bucket/cantonese-asr/cantonese_dataset.zip
unzip cantonese_dataset.zip
rm cantonese_dataset.zip
```

> Cantonse-ASR: Yu, Tiezheng, Frieske, Rita, Xu, Peng, Cahyawijaya, Samuel, Yiu, Cheuk Tung, Lovenia, Holy, Dai, Wenliang, Barezi, Elham, Chen, Qifeng, Ma, Xiaojuan, Shi, Bertram, Fung, Pascale (2022) "Automatic Speech Recognition Datasets in Cantonese: A Survey and New Dataset", 2022. Link: https://arxiv.org/pdf/2201.02419.pdf

## Cantonese Test Results on MDCC Approximately

| Model name                   | Parameters | Steps | Training Loss | Validation Loss | CER % |
| ---------------------------- | ---------- | ----- | ------------- | --------------- | ----- |
| whisper-tiny-cantonese       | 39 M       | 4400  | 0.0135        | 0.209           | 9.72  |
| whisper-base-cantonese       | 74 M       | 2400  | 0.0407        | 0.156           | 8.48  |
| whisper-small-cantonese      | 244 M      | 3600  | 0.0266        | 0.137           | 6.16  |
| whisper-small-lora-cantonese | 3.5 M      | 7600  | 0.0349        | 0.157           | 7.39  |
| whisper-large-lora-cantonese |            | 1200  | 0.0501        | 0.178           | 15.19 |
