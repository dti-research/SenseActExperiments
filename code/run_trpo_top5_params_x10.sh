#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# From: Benchmarking Reinforcement Learning Algorithms on Real-World Robots
#           https://arxiv.org/pdf/1809.07731.pdf
#
#           App. A7, Tab. 1: All hyper-parameter configurations for TRPO,
#           their value distributions and correlations with returns
#           
#           All values are drawn random (uniform distribution) from the
#           parameter ranges in App. A3.
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

for filename in experiments/ur5/trpo/*.yaml # configurations to run
do
    echo "***********************************"
    echo "* Configuration file: $filename"
    echo "***********************************"

    j=0
    while [ $j -lt 10 ] # number of tests for each hyperparameter configuration
    do
        echo " - Running test #$j for hyperparameter configuration $filename"
        python3 train.py -f $filename

        if [ $? -eq 0 ]; then
            echo " - Test #$j succeeded!"
            let j=j+1
        else
            echo " - Test #$j failed!"
        fi
        # HACK: Kill URControl process after each run
        python3 utils/ur_kill_urcontrol.py
        python3 utils/ur_reboot.py
        python3 utils/ur_send_string_command.py -ip 192.168.1.100 \
                                                -p 29999 \
                                                -c "setUserRole locked"
    done
done
