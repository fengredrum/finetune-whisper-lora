#!/usr/bin/env bash

cd ..
mkdir -p datasets/magicdata && cd datasets/magicdata
wget https://us.openslr.org/resources/68/train_set.tar.gz
tar -zxf train_set.tar.gz && rm train_set.tar.gz
wget https://us.openslr.org/resources/68/test_set.tar.gz
tar -zxf test_set.tar.gz && rm test_set.tar.gz
wget https://us.openslr.org/resources/68/dev_set.tar.gz
tar -zxf dev_set.tar.gz && rm dev_set.tar.gz
