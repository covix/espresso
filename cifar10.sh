#!/usr/bin/env bash

if [ ! -f cifar-10-python.tar.gz ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fi

tar xvf cifar-10-python.tar.gz

python3 cifar10_extract.py cifar-10-batches-py


