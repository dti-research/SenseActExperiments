#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From: Benchmarking Reinforcement Learning Algorithms on Real-World Robots
#           https://arxiv.org/pdf/1809.07731.pdf
#           Tab. A7: All hyper-parameter configurations, their value distributions and correlations with returns
#           All values are drawn random (uniform distribution) from the parameter ranges in Tab. A3.
#
# Average batch   vf-step-    δKL         γ           λ           hidden  hidden
# Return  size    size                                            layers  sizes
# 158.56  4096    0.00472     0.02437     0.96833     0.99874     2       64
# 138.58  2048    0.00475     0.01909     0.99924     0.99003     1       128
# 131.35  8192    0.00037     0.31222     0.97433     0.99647     4       64
# 123.45  4096    0.00036     0.01952     0.99799     0.92958     4       128
# 122.60  2048    0.00163     0.00510     0.96801     0.96893     4       32
# 115.51  4096    0.00926     0.01659     0.99935     0.99711     3       8
# 103.18  4096    0.00005     0.21515     0.99891     0.99880     1       8
# 100.38  8192    0.00005     0.09138     0.99677     0.99959     1       64
# 95.47   2048    0.00001     0.06088     0.98488     0.99957     2       128
# 94.16   2048    0.00770     0.02278     0.99414     0.98684     3       128
# 88.57   4096    0.00282     0.02312     0.99813     0.99964     4       16
# 65.44   512     0.00054     0.01882     0.99728     0.99420     3       32
# 63.60   8192    0.00009     0.10678     0.97415     0.99759     2       128
# 60.79   1024    0.00007     0.02759     0.99945     0.99961     3       8
# 60.51   4096    0.00222     0.00392     0.98544     0.98067     4       8
# 60.35   8192    0.00004     0.25681     0.99750     0.98955     2       128
# 59.39   1024    0.00435     0.00518     0.99516     0.99867     4       32
# 52.70   8192    0.00001     0.03385     0.99119     0.98400     4       32
# 51.44   512     0.00034     0.01319     0.97334     0.98524     4       16
# 41.05   512     0.00001     0.00351     0.99430     0.99781     3       8
# 17.14   8192    0.00023     0.01305     0.95963     0.99950     3       32
# 11.43   512     0.00251     0.00532     0.99447     0.99951     3       64
# 11.13   512     0.00003     0.00727     0.99686     0.93165     1       256
# -10.57  256     0.00065     0.04867     0.99926     0.98226     1       16
# -16.48  8192    0.00001     0.31390     0.99948     0.99204     2       16
# -19.78  512     0.00005     0.15077     0.96836     0.99944     3       64
# -32.85  256     0.00003     0.12650     0.99260     0.98021     4       128
# -43.74  8192    0.00018     0.00333     0.98940     0.97090     3       8
# -54.55  512     0.00011     0.07420     0.99402     0.90185     1       2048
# -125.13 256     0.00002     0.05471     0.99961     0.99877     4       32

# Sorted: Average return descending
hid_size=(64 128 64 128 32 8 8 64 128 128 16 32 128 8 8 128 32 32 16 8 32 64 256 16 16 64 128 8 2048 32)
num_hid_layers=(2 1 4 4 4 3 1 1 2 3 4 3 2 3 4 2 4 4 4 3 3 3 1 1 2 3 4 3 1 4)
batch_size=(4096 2048 8192 4096 2048 4096 4096 8192 2048 2048 4096 512 8192 1024 4096 8192 1024 8192 512 512 8192 512 512 256 8192 512 256 8192 512 256)
vf_stepsize=(0.00472 0.00475 0.00037 0.00036 0.00163 0.00926 0.00005 0.00005 0.00001 0.00770 0.00282 0.00054 0.00009 0.00007 0.00222 0.00004 0.00435 0.00001 0.00034 0.00001 0.00023 0.00251 0.00003 0.00065 0.00001 0.00005 0.00003 0.00018 0.00011 0.00002)
max_kl=(0.02437 0.01909 0.31222 0.01952 0.00510 0.01659 0.21515 0.09138 0.06088 0.02278 0.02312 0.01882 0.10678 0.02759 0.00392 0.25681 0.00518 0.03385 0.01319 0.00351 0.01305 0.00532 0.00727 0.04867 0.31390 0.15077 0.12650 0.00333 0.07420 0.05471)
gamma=(0.96833 0.99924 0.97433 0.99799 0.96801 0.99935 0.99891 0.99677 0.98488 0.99414 0.99813 0.99728 0.97415 0.99945 0.98544 0.99750 0.99516 0.99119 0.97334 0.99430 0.95963 0.99447 0.99686 0.99926 0.99948 0.96836 0.99260 0.98940 0.99402 0.99961)
lamda=(0.99874 0.99003 0.99647 0.92958 0.96893 0.99711 0.99880 0.99959 0.99957 0.98684 0.99964 0.99420 0.99759 0.99961 0.98067 0.98955 0.99867 0.98400 0.98524 0.99781 0.99950 0.99951 0.93165 0.98226 0.99204 0.99944 0.98021 0.97090 0.90185 0.99877)

for i in {0..29} # number of configurations to run
do
    echo "***********************************"
    echo "* Hyperparameter Configuration #$i"
    echo "* - Hidden sizes: ${hid_size[$i]}"
    echo "* - Num. hid. layers: ${num_hid_layers[$i]}"
    echo "* - Batch size: ${batch_size[$i]}"
    echo "* - VF stepsize: ${vf_stepsize[$i]}"
    echo "* - Max KL: ${max_kl[$i]}"
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
            --vf_stepsize=${vf_stepsize[$i]} \
            --max_kl=${max_kl[$i]} \
            --gamma=${gamma[$i]} \
            --lamda=${lamda[$i]} \
            --log_dir=../../logs/TRPO/RandomConf/$i
    done
done