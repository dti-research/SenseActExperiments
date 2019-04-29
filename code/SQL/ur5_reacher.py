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
from softqlearning.misc.instrument import run_sql_experiment
from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc.utils import timestamp
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SimpleSampler
from softqlearning.misc.utils import spec


from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback
from ur_setups import setup

# Directory for evaluation logs
#parser = argparse.ArgumentParser(description='Benchmarking DRL on Real World Robots')
#parser.add_argument('--log_dir', type=str, default='../../logs/SQL',
#                    help='path to put log files')
#args = parser.parse_args()

def main():
    # Set hyperparameters here
    policy_lr = 3e-4
    qf_lr = 3e-4
    discount = 0.995
    batch_size = 2048
    reward_scale = 1
    hidden_sizes = (32, 32)
    
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
            rllab_box=True,
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

    # Create replay buffer
    pool = SimpleReplayBuffer(
        env_spec=spec(env),
        max_replay_buffer_size=1e6
    )

    sampler = SimpleSampler(
        max_path_length=1000,
        min_pool_size=1000,
        batch_size=batch_size
    )

    base_kwargs = dict(
        epoch_length=150000,
        n_epochs=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler
    )

    # Policy function
    policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=hidden_sizes)

    # Q-function
    qf = NNQFunction(env_spec=env.spec, hidden_sizes=hidden_sizes)

    # Create and start logging process
    #log_running = Value('i', 1)
    # Manager to share data between log process and main process
    #shared_returns = Manager().dict({'write_lock': False,
    #                                 'episodic_returns': [],
    #                                 'episodic_lengths': [], }) 
    
    # Spawn logging process
    #pp = Process(target=log_function, args=(env, batch_size, log_running, args.log_dir))
    #pp.start()

    # Create callback function for logging data from baselines TRPO learn
    #kindred_callback = create_callback(shared_returns)

    # Train softqlearning SQL
    algo = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool, 
        qf=qf,
        policy=policy,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=16,
        kernel_update_ratio=0.5,
        value_n_particles=16,
        td_target_update_interval=1000,
        qf_lr=qf_lr,
        policy_lr=policy_lr,
        discount=discount,
        reward_scale=reward_scale,
        save_full_state=False
        )
    algo.train()

    # Safely terminate plotter process
    #log_running.value = 0  # shutdown plotting process
    #time.sleep(2)
    #pp.join()

    env.close()

#def log_function(env, batch_size, shared_returns, log_running, log_dir): 
    """ Process for logging all data generated during runtime.
    Args:
	env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns: A manager dictionary object containing 'episodic returns' and 'episodic lengths'
        log_running: A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    

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
                        
    file.close()"""

if __name__ == '__main__':
    main()
