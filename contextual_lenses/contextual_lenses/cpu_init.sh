#!/bin/bash

pip install --upgrade pip
pip install --upgrade jax jaxlib flax
export TF_FORCE_GPU_ALLOW_GROWTH=true