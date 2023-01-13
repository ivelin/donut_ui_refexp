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

virtualenv -p python3.8 .
source ./bin/activate
pip install -r requirements.txt

git pull
git config --global credential.helper "cache --timeout=3600"
wandb login
huggingface-cli login