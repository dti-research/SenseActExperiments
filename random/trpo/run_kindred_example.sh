#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From: Benchmarking Reinforcement Learning Algorithms on Real-World Robots
#           https://arxiv.org/pdf/1809.07731.pdf
# Parameters taken from: https://github.com/kindredresearch/SenseAct/blob/master/examples/advanced/ur5_reacher.py
#    WARNING:   These parameters are not documented nor guaranteed to converge to any form for useful solution!
#

python ur5_reacher.py \
    --hid_size=32 \
    --num_hid_layers=2 \
    --batch_size=2048 \
    --vf_stepsize=0.001 \
    --max_kl=0.05 \
    --gamma=0.995 \
    --lamda=0.995 \
    --log_dir=logs/TRPO/KindredExample
