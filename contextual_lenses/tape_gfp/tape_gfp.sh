#!/bin/bash

git clone https://github.com/songlab-cal/tape
pip install -r tape/requirements.txt
cd tape
wget http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz
tar xzf fluorescence.tar.gz