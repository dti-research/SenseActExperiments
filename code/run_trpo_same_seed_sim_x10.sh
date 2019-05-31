#!/bin/bash

# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

robot_ip=10.224.60.159
robot_port=29999
robot_sim_username=ur


python3 utils/ur_kill_urcontrol.py --robot-ip $robot_ip \
                                   --username $robot_sim_username \
                                   --sim True
python3 utils/ur_reboot.py --robot-ip $robot_ip \
                           --username $robot_sim_username \
                           --sim True
python3 utils/ur_send_string_command.py -ip $robot_ip \
                                        -p $robot_port \
                                        -c "setUserRole locked"

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
    python3 utils/ur_kill_urcontrol.py --robot-ip $robot_ip \
                                       --username $robot_sim_username \
                                       --sim True
    python3 utils/ur_reboot.py --robot-ip $robot_ip \
                               --username $robot_sim_username \
                               --sim True
    python3 utils/ur_send_string_command.py -ip $robot_ip \
                                            -p $robot_port \
                                            -c "setUserRole locked"
done
