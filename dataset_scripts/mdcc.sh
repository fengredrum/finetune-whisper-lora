#!/usr/bin/env bash

cd ..
mkdir -p datasets/mdcc && cd datasets/mdcc
wget https://storage.googleapis.com/samcah-bucket/cantonese-asr/cantonese_dataset.zip
unzip cantonese_dataset.zip
rm cantonese_dataset.zip
