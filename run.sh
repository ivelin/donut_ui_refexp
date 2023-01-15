#!/bin/bash


# Check TPU readiness
# https://cloud.google.com/tpu/docs/users-guide-tpu-vm
# gcloud compute tpus tpu-vm list --zone=us-central1-f
# gcloud compute tpus tpu-vm describe  tpu2-us-central1-f --zone=us-central1-f

# https://cloud.google.com/tpu/docs/supported-tpu-configurations#pytorch


# Make sure to run setup.sh first!

# virtualenv -p python3.8 venv
# source ./venv/bin/activate

# Restart TPU service
# python3 -m torch_xla.core.xrt_run_server --port 51011 --restart

# See docs for accelerator and devices params you should be using
# Defaults to v2-8 Google TPUs:
# https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-tpu-training.html
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1


python3 -m ui_refexp.train --accelerator='tpu' --devices=8
