#!/bin/bash

pip install --upgrade pip

PYTHON_VERSION=cp37
CUDA_VERSION=cuda101
PLATFORM=linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'

pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.48-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax flax

export TF_FORCE_GPU_ALLOW_GROWTH=true