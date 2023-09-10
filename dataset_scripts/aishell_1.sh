#!/usr/bin/env bash

cd ..
mkdir -p datasets/aishell_1 && cd datasets/aishell_1
wget https://us.openslr.org/resources/33/data_aishell.tgz
tar -zxf data_aishell.tgz && rm data_aishell.tgz
cd data_aishell/wav
for wav in ./*.tar.gz; do
    echo "Extracting wav from $wav"
    tar -zxf $wav && rm $wav
done
