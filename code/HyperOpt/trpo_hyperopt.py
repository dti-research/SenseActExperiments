# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function
import time
import copy
import tensorflow as tf
import numpy as np
import os
import io
import argparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample

import baselines.common.tf_util as U
from multiprocessing import Process, Value, Manager
from baselines.trpo_mpi.trpo_mpi import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback
from ur_setups import setup

# Argparser used for log directories
parser = argparse.ArgumentParser(description='Hyperparameter optimization')
parser.add_argument('--log_dir', type=str, default='../../logs/HyperOpt/TRPO',
                    help='path to put log files')
args = parser.parse_args()

# File names
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
time_now = time.time()
filename = os.path.join(args.log_dir, str(time_now) + '_trpo_logs')
hp_filename = os.path.join(args.log_dir, str(time_now) + 'hp_trpo.csv')

# Define the search space of each hyperparameter for the Bayesian hyperparameter optimization
space = { 
        'num_hid_layers': hp.choice('num_hid_layers', [
                                       {'num_hid_layers': 1, 
                                               'hid_size': hp.choice('1_hid_size', [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])},#depending on # hidden layers
                                        {'num_hid_layers': 2,
                                               'hid_size': hp.choice('2_hid_size', [8, 16, 32, 64, 128, 256])},
                                        {'num_hid_layers': 3,
                                               'hid_size': hp.choice('3_hid_size', [8, 16, 32, 64, 128])},
                                        {'num_hid_layers': 4,
                                               'hid_size': hp.choice('4_hid_size', [8, 16, 32, 64, 128])}]),
        'timesteps_per_batch': hp.choice('timesteps_per_batch', [256, 512, 1024, 2048, 4096, 8192]),
        'vf_stepsize': hp.loguniform('vf_stepsize', -5, -2),
        'max_kl' : hp.loguniform('max_kl', -2.5, -0.5),
        'gamma': hp.uniform('gamma', (1-(1/((0.1)*4))), (1-(1/((31.62)*4)))), #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
        'lam': hp.uniform('lam', (1-(1/((0.1)*4))), (1-(1/((31.62)*4)))) #4:T. Remember to change this if code is altered. -1:T/tau. tau=0.04=dt
}

def main():
    global ITERATION

    ITERATION = 0
    max_evaluations = 10

    file = open(hp_filename, 'w')
    file.write('iteration,loss,hidden-layers,hidden-sizes,batch-size,stepsize,kl-divergence,gamma,lambda\n')
    file.close()

    # Keep track of results
    bayes_trials = Trials()

    # Optimization algorithm
    best = fmin(fn = objective,
		space = space, 
		algo = tpe.suggest, 
		max_evals = max_evaluations, 
		trials = bayes_trials)
    
    
    print(space) #debugging , delete
    print(best)


# Objective function used for Bayesian hyperparameter tuning
def objective(hyperparams):
    """Objective function for TRPO hyperparameter tuning"""

    # Keep track of evaluations
    global ITERATION
    ITERATION += 1

    # For logging rewards
    returns = []
    shared_returns = Manager().dict({'write_lock': False,
                                     'episodic_returns': [],
                                     'episodic_lengths': [],
                                     'rets': []}) # A manager dictionary object containing `episodic returns` and `episodic lengths`

    # Get all the current hyperparameter values
    hid_size = hyperparams['num_hid_layers'].get('hid_size') # Need to retrieve like this, because it is nested in dictionary

    hyperparams['num_hid_layers'] = hyperparams['num_hid_layers']['num_hid_layers']
    hyperparams['hid_size'] = hid_size
    hyperparams['timesteps_per_batch'] = hyperparams['timesteps_per_batch']

    for parameter_name in ['vf_stepsize', 'max_kl', 'gamma', 'lam']:
        hyperparams[parameter_name] = float(hyperparams[parameter_name])

    # Run the TRPO algorithm
    pp2 = Process(target=run_trpo, args=(hyperparams, shared_returns)) # Changed parameters here, not sure if it still works, maybe it doesn't have accessto needed data
    pp2.start()

    time.sleep(2)
    pp2.join()

    copied_returns = copy.deepcopy(shared_returns)
    returns = copied_returns['episodic_returns']

    # Extract the highest reward
    best_score = max(returns)

    # Transform to loss, to obtain a function to minimize
    loss = 1 - best_score

    # Log the current hyperparameter setting, iteration and loss in a csv-file
    file = open(hp_filename, 'a')
    file.write(str(ITERATION) + ',' + str(loss) + ',' + str(hyperparams['num_hid_layers']) + ',' + str(hyperparams['hid_size']) + 
                ',' + str(hyperparams['timesteps_per_batch']) + ',' + str(hyperparams['vf_stepsize']) + ',' + 
                str(hyperparams['max_kl']) + ',' + str(hyperparams['gamma']) + ',' + str(hyperparams['lam']) + '\n')
    file.close()

    # Return loss, current hyperparameter configuration, iteration and key indicating if evaluation was succesful
    return {'loss': loss, 'hyperparams': hyperparams, 'iteration': ITERATION, 'status': STATUS_OK}


# Function for running TRPO
def run_trpo(hyperparams, shared_returns):
    """All the functionality implemented to run the RL algorithm on the UR robot"""

    # Use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

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
        return MlpPolicy(name=name, 
                         ob_space=ob_space, 
                         ac_space=ac_space,
                         hid_size=hyperparams['hid_size'], 
                         num_hid_layers=hyperparams['num_hid_layers']
                        )

    # Create and start logging process
    log_running = Value('i', 1) # flag

    # Spawn logging process
    pp = Process(target=log_function, args=(env, hyperparams['timesteps_per_batch'], shared_returns, log_running)) 
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns)

    # Train baselines TRPO
    learn(env, policy_fn,
          max_timesteps=2000,
          timesteps_per_batch=hyperparams['timesteps_per_batch'],
          max_kl=hyperparams['max_kl'],
          cg_iters=10,
          cg_damping=0.1,
          vf_iters=5,
          vf_stepsize=hyperparams['vf_stepsize'],
          gamma=hyperparams['gamma'],
          lam=hyperparams['lam'],
          callback=kindred_callback
          )

    # Safely terminate plotter process
    log_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    sess.close()
    env.close()
    #tf.reset_default_graph()

# Function for logging relevant values to csv-file
def log_function(env, batch_size, shared_returns, log_running): 
    """ Process for logging all data generated during runtime.
    Args:
	env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        log_running: A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    """
    global ITERATION
    
    old_size = len(shared_returns['episodic_returns']) # Initialize variable old_size with the lenght of mystical array episodic_returns from helper.py
    time.sleep(5.0) # Process sleep for 5 seconds

    file = open((filename + str(ITERATION) + '.csv'), 'a')

    file.write('Episode,Step,Reward,X-Target,Y-Target,Z-Target,X-Current,Y-Current,Z-Current \n') # Header with names for all values logged
    
    while log_running.value:
        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        copied_returns = copy.deepcopy(shared_returns)

        # Write current values to file
        if not copied_returns['write_lock'] and  len(copied_returns['episodic_returns']) > old_size:
            old_size = len(copied_returns['episodic_returns'])
            episode = len(copied_returns['episodic_lengths'])

            file.write(str(episode) + ',' + str(int(episode/0.04)) + ',' + str(copied_returns['episodic_returns'][-1]) + 
                        ',' + str(env._x_target_[2]) + ',' + str(env._x_target_[1])  + ',' + str(env._x_target_[0])
                        + ',' + str(env._x_[2]) + ',' + str(env._x_[1]) + ',' + str(env._x_[0]) + '\n')

        """# Calculate rolling average of rewards
            #returns = np.array(copied_returns['episodic_returns'])
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
                rets = []

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))"""
        
    file.close()

if __name__ == '__main__':
    main()