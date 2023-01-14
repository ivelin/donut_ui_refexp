#!/bin/bash

# Make sure to run setup.sh first!

virtualenv -p python3.8 .
source ./bin/activate venv

# See docs for accelerator and devices params you should be using
# Defaults to v2-8 Google TPUs:
# https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-tpu-training.html
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=0


python3 -m ui_refexp.train --accelerator='tpu' --devices=1
