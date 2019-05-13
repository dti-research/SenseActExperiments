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
# Average   batch   step-size   opt.        γ           λ           hidden  hidden
# Return    size                batch size                          layers  sizes
# 176.62    512     0.00005     16          0.96836     0.99944     3       64
# 150.25    256     0.00050     64          0.99926     0.98226     1       16
# 137.92    512     0.00011     8           0.99402     0.90185     1       2048
# 137.26    2048    0.00163     1024        0.96801     0.96893     4       32
# 136.09    2048    0.00280     32          0.99924     0.99003     1       128
# 128.34    4096    0.00036     64          0.99799     0.92958     4       128
# 118.77    512     0.00003     32          0.99686     0.93165     1       256
# 112.48    4096    0.00941     1024        0.98544     0.98067     4       8
# 110.01    4096    0.00080     8           0.99935     0.99711     3       8
# 107.47    4096    0.00267     4096        0.96833     0.99874     2       64
# 95.62     8192    0.00226     32          0.97433     0.99647     4       64
# 82.21     8192    0.00037     16          0.99119     0.98400     4       32
# 78.97     512     0.00090     128         0.99430     0.99781     3       8
# 73.33     4096    0.00079     32          0.99813     0.99964     4       16
# 73.17     256     0.00003     16          0.99260     0.98021     4       128
# 56.25     8192    0.00987     32          0.99948     0.99204     2       16
# 49.02     8192    0.00019     64          0.99677     0.99959     1       64
# 29.70     1024    0.00039     256         0.99945     0.99961     3       8
# 25.94     8192    0.00362     32          0.97415     0.99759     2       128
# 18.64     4096    0.00061     512         0.99891     0.99880     1       8
# -1.68     8192    0.00006     2048        0.99750     0.98955     2       128
# -20.53    8192    0.00087     2048        0.98940     0.97090     3       8
# -31.58    1024    0.00315     256         0.99516     0.99867     4       32
# -49.32    512     0.00680     128         0.99728     0.99420     3       32
# -62.46    2048    0.00002     2048        0.99414     0.98684     3       128
# -64.50    2048    0.00002     1024        0.98488     0.99957     2       128
# -79.45    8192    0.00002     256         0.95963     0.99950     3       32
# -84.29    512     0.00002     256         0.97334     0.98524     4       16
# -94.34    512     0.00645     32          0.99447     0.99951     3       64
# -134.43   256     0.00689     128         0.99961     0.99877     4       32

# Sorted: Average return descending
hid_size=(64 16 2048 32 128 128 256 8 8 64 64 32 8 16 128 16 64 8 128 8 128 8 32 32 128 128 32 16 64 32)
num_hid_layers=(3 1 1 4 1 4 1 4 3 2 4 4 3 4 4 2 1 3 2 1 2 3 4 3 3 2 3 4 3 4)
batch_size=(512 256 512 2048 2048 4096 512 4096 4096 4096 8192 8192 512 4096 256 8192 8192 1024 8192 4096 8192 8192 1024 512 2048 2048 8192 512 512 256)
step_size=(0.00005 0.00050 0.00011 0.00163 0.00280 0.00036 0.00003 0.00941 0.00080 0.00267 0.00226 0.00037 0.00090 0.00079 0.00003 0.00987 0.00019 0.00039 0.00362 0.00061 0.00006 0.00087 0.00315 0.00680 0.00002 0.00002 0.00002 0.00002 0.00645 0.00689)
opt_batch_size=(16 64 8 1024 32 64 32 1024 8 4096 32 16 128 32 16 32 64 256 32 512 2048 2048 256 128 2048 1024 256 256 32 128)
gamma=(0.96836 0.99926 0.99402 0.96801 0.99924 0.99799 0.99686 0.98544 0.99935 0.96833 0.97433 0.99119 0.99430 0.99813 0.99260 0.99948 0.99677 0.99945 0.97415 0.99891 0.99750 0.98940 0.99516 0.99728 0.99414 0.98488 0.95963 0.97334 0.99447 0.99961)
lamda=(0.99944 0.98226 0.90185 0.96893 0.99003 0.92958 0.93165 0.98067 0.99711 0.99874 0.99647 0.98400 0.99781 0.99964 0.98021 0.99204 0.99959 0.99961 0.99759 0.99880 0.98955 0.97090 0.99867 0.99420 0.98684 0.99957 0.99950 0.98524 0.99951 0.99877)

for i in {0..29} # number of configurations to run
do
    echo "***********************************"
    echo "* Hyperparameter Configuration #$i"
    echo "* - Hidden sizes: ${hid_size[$i]}"
    echo "* - Num. hid. layers: ${num_hid_layers[$i]}"
    echo "* - Batch size: ${batch_size[$i]}"
    echo "* - Step-size: ${step_size[$i]}"
    echo "* - Optimiser Batch Size: ${opt_batch_size[$i]}"
    echo "* - Gamma: ${gamma[$i]}"
    echo "* - Lamda: ${lamda[$i]}"
    echo "***********************************"

    for j in {0..9} # number of tests for each hyperparameter configuration
    do
        echo " - Running test #$j for hyperparameter configuration #$i"
        
        python ur5_reacher.py \
            --hid_size=${hid_size[$i]} \
            --num_hid_layers=${num_hid_layers[$i]} \
            --batch_size=${batch_size[$i]} \
            --step_size=${step_size[$i]} \
            --opt_batch_size=${opt_batch_size[$i]} \
            --gamma=${gamma[$i]} \
            --lamda=${lamda[$i]} \
            --log_dir=logs/PPO/RandomConf/$i
    done
done