#!/bin/bash

# Copyright 2022 Ivelin Ivanov and Guardian UI LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x

# virtualenv -p python3.8 venv
# source ./venv/bin/activate
# pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip3 install -r requirements.txt


# For GCP TPU XLA lib support
# wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# git pull
git config --global credential.helper store
python3 -m wandb login
# huggingface-cli login
python3 -m huggingface_hub.commands.huggingface_cli login
