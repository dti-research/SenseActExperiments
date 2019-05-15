# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import time
import copy
import numpy as np
import os
import io
import datetime
import argparse

import baselines.common.tf_util as U
from multiprocessing import Process, Value, Manager
from baselines.trpo_mpi.trpo_mpi import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback
from ur_setups import setup

parser = argparse.ArgumentParser(description='Benchmarking DRL on Real World Robots')
parser.add_argument('--max_timesteps', type=int, default=150000,
                    help='number of timesteps used in the learning phase (default: 150000)')
parser.add_argument('--hid_size', type=int, default=64,
                    help='size of hidden layers (default: 64)')
parser.add_argument('--num_hid_layers', type=int, default=2,
                    help='number of hidden layers (default: 2)')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='input batch size for training (default: 4096)')
parser.add_argument('--vf_stepsize', type=float, default=0.00472,
                    help='stepsize (default: 0.00472)')
parser.add_argument('--max_kl', type=float, default=0.02437,
                    help='max_kl (default: 0.02437)')
parser.add_argument('--gamma', type=float, default=0.96833,
                    help='gamma (default: 0.96833)')
parser.add_argument('--lamda', type=float, default=0.99874,
                    help='lamda (default: 0.99874)')
parser.add_argument('--log_dir', type=str, default="../../logs/TRPO",
                    help='path to put log files')
args = parser.parse_args()

def main():
    # Set hyperparameters here
    max_timesteps       = args.max_timesteps
    hid_size            = args.hid_size
    num_hid_layers      = args.num_hid_layers
    timesteps_per_batch = args.batch_size
    vf_stepsize         = args.vf_stepsize
    max_kl              = args.max_kl
    gamma               = args.gamma
    lam                 = args.lamda

    # WARNING: Hardcoded values from Kindred's example with no explanation.
    #  https://github.com/kindredresearch/SenseAct/blob/master/examples/advanced/ur5_reacher.py#L78
    #  - No explanation found in paper.
    #  - No explanation from Kindred by mail correspondance.
    vf_iters   = 5
    cg_iters   = 10
    cg_damping = 0.1
    
    # Use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Create UR5 Reacher2D environment
    env = ReacherEnv(
            setup=setup,
            host=None,
            dof=2,
            control_type='velocity',
            target_type='position',
            reset_type='zero',
            reward_type='precision',
            derivative_type='none',
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.3,
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.04,
            run_mode='multiprocess',
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

    # Create and start logging process
    log_running = Value('i', 1)
    # Manager to share data between log process and main process
    shared_returns = Manager().dict({'write_lock': False,
                                     'episodic_returns': [],
                                     'episodic_lengths': [], }) 
    
    # Spawn logging process
    pp = Process(target=log_function, args=(env, timesteps_per_batch, shared_returns, log_running, args.log_dir))
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns)

    # Train baselines TRPO
    learn(env, policy_fn,
          max_timesteps=max_timesteps,
          timesteps_per_batch=timesteps_per_batch,
          max_kl=max_kl,
          cg_iters=cg_iters,
          cg_damping=cg_damping,
          vf_iters=vf_iters,
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

def log_function(env, batch_size, shared_returns, log_running, log_dir): 
    """ Process for logging all data generated during runtime.
    Args:
	env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns: A manager dictionary object containing 'episodic returns' and 'episodic lengths'
        log_running: A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    """

    old_size = len(shared_returns['episodic_returns']) 
    time.sleep(5.0)
    rets = []

    # Create logs directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    time_now = time.time()
    file = open(os.path.join(os.path.join(log_dir, str(time_now)+'.csv')), 'a')

    file.write('Episode,Step,Reward,X-Target,Y-Target,Z-Target,X-Current,Y-Current,Z-Current \n') # Header with names for all values logged

    while log_running.value:
        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        copied_returns = copy.deepcopy(shared_returns)
        episode = len(copied_returns['episodic_lengths'])

        # Write current values to file
        if not copied_returns['write_lock'] and  len(copied_returns['episodic_returns']) > old_size:
            returns = np.array(copied_returns['episodic_returns'])
            old_size = len(copied_returns['episodic_returns'])

            if len(rets):
                file.write(str(episode) + ',' + str(int(episode*100)) + ',' + str(rets[-1]) + 
                           ',' + str(env._x_target_[2]) + ',' + str(env._x_target_[1])  + ',' + str(env._x_target_[0])
                           + ',' + str(env._x_[2]) + ',' + str(env._x_[1]) + ',' + str(env._x_[0]) + '\n')
            else:
                file.write(str(episode) + ',' + str(int(episode*100)) + ',' + 'NaN' + 
                           ',' + str(env._x_target_[2]) + ',' + str(env._x_target_[1])  + ',' + str(env._x_target_[0])
                           + ',' + str(env._x_[2]) + ',' + str(env._x_[1]) + ',' + str(env._x_[0]) + '\n')
            
            # Calculate rolling average of rewards
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))
                        
    file.close()

if __name__ == '__main__':
    main()
