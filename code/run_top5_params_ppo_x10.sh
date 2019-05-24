#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From: Benchmarking Reinforcement Learning Algorithms on Real-World Robots
#           https://arxiv.org/pdf/1809.07731.pdf
#
#           App. A7, Tab. 3: All hyper-parameter configurations for PPO,
#           their value distributions and correlations with returns
#           
#           All values are drawn random (uniform distribution) from the
#           parameter ranges in App. A3.
#
# Average   batch   step-size   opt.        γ       λ       hidden hidden
# Return    size                batch size                  layers sizes
# 176.62    512     0.00005     16          0.96836 0.99944 3       64
# 150.25    256     0.00050     64          0.99926 0.98226 1       16
# 137.92    512     0.00011     8           0.99402 0.90185 1       2048
# 137.26    2048    0.00163     1024        0.96801 0.96893 4       32
# 136.09    2048    0.00280     32          0.99924 0.99003 1       128
# 128.34    4096    0.00036     64          0.99799 0.92958 4       128
# 118.77    512     0.00003     32          0.99686 0.93165 1       256
# 112.48    4096    0.00941     1024        0.98544 0.98067 4       8
# 110.01    4096    0.00080     8           0.99935 0.99711 3       8
# 107.47    4096    0.00267     4096        0.96833 0.99874 2       64
# 95.62     8192    0.00226     32          0.97433 0.99647 4       64
# 82.21     8192    0.00037     16          0.99119 0.98400 4       32
# 78.97     512     0.00090     128         0.99430 0.99781 3       8
# 73.33     4096    0.00079     32          0.99813 0.99964 4       16
# 73.17     256     0.00003     16          0.99260 0.98021 4       128
# 56.25     8192    0.00987     32          0.99948 0.99204 2       16
# 49.02     8192    0.00019     64          0.99677 0.99959 1       64
# 29.70     1024    0.00039     256         0.99945 0.99961 3       8
# 25.94     8192    0.00362     32          0.97415 0.99759 2       128
# 18.64     4096    0.00061     512         0.99891 0.99880 1       8
# -1.68     8192    0.00006     2048        0.99750 0.98955 2       128
# -20.53    8192    0.00087     2048        0.98940 0.97090 3       8
# -31.58    1024    0.00315     256         0.99516 0.99867 4       32
# -49.32    512     0.00680     128         0.99728 0.99420 3       32
# -62.46    2048    0.00002     2048        0.99414 0.98684 3       128
# -64.50    2048    0.00002     1024        0.98488 0.99957 2       128
# -79.45    8192    0.00002     256         0.95963 0.99950 3       32
# -84.29    512     0.00002     256         0.97334 0.98524 4       16
# -94.34    512     0.00645     32          0.99447 0.99951 3       64
# -134.43   256     0.00689     128         0.99961 0.99877 4       32

for filename in experiments/ur5/ppo/*.yaml # configurations to run
do
    echo "***********************************"
    echo "* Configuration file: $filename"
    echo "***********************************"

    j=0
    #for j in {0..9} # number of tests for each hyperparameter configuration
    while [ $j -lt 10 ]
    do
        echo " - Running test #$j for hyperparameter configuration $filename"
        python3 train.py -f $filename

        if [ $? -eq 0 ]; then
            echo " - Test #$j succeeded!"
            let j=j+1
        else
            echo " - Test #$j failed!"
        fi
        python3 utils/ur_reboot.py
    done
done
