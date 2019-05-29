#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

python3 utils/ur_kill_urcontrol.py --robot-ip 10.224.60.159 --username ur
python3 utils/ur_reboot.py --robot-ip 10.224.60.159 --username ur
python3 utils/ur_lock.py --robot-ip 10.224.60.159

file=experiments/ur5/trpo_ursim_same_seed.yaml
j=0
while [ $j -lt 10 ] # number of tests for each hyperparameter configuration
do
    echo " - Running test #$j for hyperparameter configuration $file"
    python3 train.py -f $file
    if [ $? -eq 0 ]; then
        echo " - Test #$j succeeded!"
        let j=j+1
    else
        echo " - Test #$j failed!"
    fi
    # HACK: Kill URControl process after each run
    python3 utils/ur_kill_urcontrol.py --robot-ip 10.224.60.159 --username ur
    python3 utils/ur_reboot.py --robot-ip 10.224.60.159 --username ur
    python3 utils/ur_send_script_command.py -ip 10.224.60.159 \
                                            -p 29999 \
                                            -c "setUserRole locked"
done
