#!/usr/bin/env bash

cd ..
mkdir -p datasets/thchs_30 && cd datasets/thchs_30
wget https://us.openslr.org/resources/18/data_thchs30.tgz
tar -zxf data_thchs30.tgz && rm data_thchs30.tgz
