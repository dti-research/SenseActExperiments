# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import copy
import numpy as np
import io
import datetime
#import csv

import baselines.common.tf_util as U
from multiprocessing import Process, Value, Manager
from baselines.trpo_mpi.trpo_mpi import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback

setup = {
                  'host': '192.168.1.100', # put UR5 Controller address here (192.168.2.152) (10.44.60.122)
                  'end_effector_low': np.array([-0.2, -0.3, 0.5]),
                  'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'angles_low':np.pi/180 * np.array(
                      [ 60,
                       -180,#-180
                       -120,
                       -50,
                        50,
                        50
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 90,
                       -60,
                        130,
                        25,
                        120,
                        175
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.array([ 1.58724391, -2.4, 1.5, -0.71790582, 1.63685572, 1.00910473]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )

}

def main():

    # Set hyperparameters here
    hid_size = 32              # 2^[3,X] where X is ?
    num_hid_layers = 2         # [1,4]
    timesteps_per_batch = 2048 # 2^[8,13]
    vf_stepsize = 0.001        # 10^[-5,-2]
    max_kl = 0.05              # 10^[-2.5,-0.5]
    cg_iters = 10              # I don't know what this is in the table from SenseAct paper b, see trpo_mpi
    cg_damping = 0.1           # I don't know what this is either see trpo_mpi
    gamma = 0.995              # 1 - (1/(c_gamma*N)), where N = T/tau, T is the total length of an ep in time (?) and tau is the action cycle time
    lam = 0.995                # 1 - (1/(c_lambda*N)), where N = T/tau, T is the total length of an ep in time (?) and tau is the action cycle time    

    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Create UR5 Reacher2D environment
    dt = 0.04 # Used to log data, therefore it is defined here

    env = ReacherEnv(
            setup=setup,
            host=None,
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.3,
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.start()
    # Create baselines TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers)

    # CHANGES HERE!! Create and start logging process  CHANGES HERE!!
    log_running = Value('i', 1) # flag
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], }) # A manager dictionary object containing `episodic returns` and `episodic lengths`
    # Spawn logging process
    pp = Process(target=log_to_csv, args=(env, 2048, shared_returns, log_running, dt))
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns)

    # Train baselines TRPO
    learn(env, policy_fn,
          max_timesteps=150000,
          timesteps_per_batch=timesteps_per_batch,
          max_kl=max_kl,
          cg_iters=cg_iters,
          cg_damping=cg_damping,
          vf_iters=5,
          vf_stepsize=vf_stepsize,
          gamma=gamma,
          lam=lam,
          callback=kindred_callback
          )

    # Safely terminate plotter process
    log_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()

def log_to_csv(env, batch_size, shared_returns, log_running, dt): 
    """ Process for logging all data generated during runtime.
    Args:
	env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        log_running: A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    """

    file = open("trpo_logs.csv", "w")

    time_now = time.time()

    file.write("Time,Reward,X-Target,Y-Target,Z-Target,X-Current,Y-Current,Z-Current \n") # Header with names for all values logged
    step = 0

    while log_running.value:
        if time.time() - time_now >= dt:
            time_now = time.time()
            file.write(str(step) + "," + str(env._reward_.value) + "," + str(env._x_target_[2]) + "," + str(env._x_target_[1]) + "," + str(env._x_target_[0]) + "," + 			str(env._x_[2]) + "," + str(env._x_[1]) + "," + str(env._x_[0]) + "\n")
            i += 1

    file.close()


if __name__ == '__main__':
    main()