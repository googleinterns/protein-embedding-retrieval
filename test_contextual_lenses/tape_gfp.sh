#!/bin/bash

git clone https://github.com/songlab-cal/tape
python3.7 -m pip install -q -r tape/requirements.txt
cd tape
wget http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz
tar xzf fluorescence.tar.gz