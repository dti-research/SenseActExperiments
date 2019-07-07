#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

i=0
seed=20
algorithm=trpo
values=(158.56 138.58 131.35 123.45 122.60)
path=../code/artifacts/logs/$algorithm

for d in $path/*
do
    if [ -d "${d}" ]; then
        echo " - Evaluating configuration #$i for $algorithm"

        python3 dist_fitter.py -p $d \
                            --algorithm $algorithm \
                            --configuration $i \
                            --output-path plots/fitted_dists/seed$seed/ \
                            --value=${values[$i]} \
                            --seed=$seed
        let i=i+1
    fi
done

