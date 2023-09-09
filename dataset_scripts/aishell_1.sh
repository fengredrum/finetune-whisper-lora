#!/usr/bin/env bash

cd ./datasets/aishell-1/data_aishell/wav
for wav in ./*.tar.gz; do
    echo "Extracting wav from $wav"
    tar -zxf $wav && rm $wav
done