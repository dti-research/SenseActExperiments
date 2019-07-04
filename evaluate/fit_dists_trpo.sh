#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

i=0
seed=20
algorithm=trpo
path=../code/artifacts/logs/$algorithm

for d in $path/*
do
    if [ -d "${d}" ]; then
        echo " - Evaluating configuration #$i for $algorithm"

        python3 dist_fitter.py -p $d \
                            --algorithm $algorithm \
                            --configuration $i \
                            --output-path plots/fitted_dists/seed$seed/ \
                            --seed=$seed
        let i=i+1
    fi
done

