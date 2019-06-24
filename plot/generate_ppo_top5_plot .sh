#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


python plot_w_std_err.py -p ../code/artifacts/logs/ppo \
                          --output-filename ppo.svg \
                          --output-path plots/