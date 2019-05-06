# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an experiment."""

import time
import copy
import numpy as np
import os
import io
import datetime
import logging
import sys
import yaml
import importlib

import baselines.common.tf_util as U
from multiprocessing import Process, Value, Manager
from senseact.utils import tf_set_seeds, NormalizedEnv

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(yaml_filepath):
    """Example."""
    # Load yaml configuration file
    cfg = load_cfg(yaml_filepath)

    # Print the configuration - just to make sure that you loaded what you wanted to load
    logging.debug(yaml.dump(cfg, default_flow_style=False))

    

    # HACK: Should not be in this file!
    # Train the agent
    train(cfg)

def train(cfg):
    # Load artifact path
    artifact_path = cfg['train']['artifacts_path']
    logging.debug(artifact_path)
    if not os.path.exists(artifact_path): os.makedirs(artifact_path)

    # Load the RL Environment
    env_module = importlib.import_module(cfg['environment']['module'])
    env_class = getattr(env_module, cfg['environment']['class'])
    logging.debug(env_class)

    # Use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Create UR5 Reacher2D environment
    env = env_class(
            setup = setup,
            host                  = cfg['environment']['parameters']['host'],
            dof                   = cfg['environment']['parameters']['dof'],
            control_type          = cfg['environment']['parameters']['control_type'],
            target_type           = cfg['environment']['parameters']['target_type'],
            reset_type            = cfg['environment']['parameters']['reset_type'],
            reward_type           = cfg['environment']['parameters']['reward_type'],
            derivative_type       = cfg['environment']['parameters']['derivative_type'],
            deriv_action_max      = cfg['environment']['parameters']['deriv_action_max'],
            first_deriv_max       = cfg['environment']['parameters']['first_deriv_max'],
            accel_max             = cfg['environment']['parameters']['accel_max'],
            speed_max             = cfg['environment']['parameters']['speed_max'],
            speedj_a              = cfg['environment']['parameters']['speedj_a'],
            episode_length_time   = cfg['environment']['parameters']['episode_length_time'],
            episode_length_step   = cfg['environment']['parameters']['episode_length_step'],
            actuation_sync_period = cfg['environment']['parameters']['actuation_sync_period'],
            dt                    = cfg['environment']['parameters']['dt'],
            run_mode              = cfg['environment']['parameters']['run_mode'],
            rllab_box             = cfg['environment']['parameters']['rllab_box'],
            movej_t               = cfg['environment']['parameters']['movej_t'],
            delay                 = cfg['environment']['parameters']['delay'],
            random_state          = rand_state
        )
    env = NormalizedEnv(env)

    # Start environment processes
    env.start()

    # Create baselines TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()
    policy_fn_module = importlib.import_module(cfg['model']['module'])
    policy_fn_class = getattr(policy_fn_module, cfg['model']['class'])
    logging.debug(policy_fn_class)

    def policy_fn(name, ob_space, ac_space):
        return policy_fn_class(name           = name,
                               ob_space       = ob_space,
                               ac_space       = ac_space,
                               hid_size       = cfg['algorithm']['parameters']['hid_size'],
                               num_hid_layers = cfg['algorithm']['parameters']['num_hid_layers'])
    
    # Create and start logging process
    log_running = Value('i', 1)
    # Manager to share data between log process and main process
    shared_returns = Manager().dict({'write_lock': False,
                                     'episodic_returns': [],
                                     'episodic_lengths': [], }) 
    
    # Spawn logging process
    pp = Process(target=log_function,
                 args=(env,
                       cfg['algorithm']['parameters']['timesteps_per_batch'],
                       shared_returns,
                       log_running,
                       artifact_path
                       )
                )
    pp.start()

    """
    # Create callback function for logging data from learn
    kindred_callback = create_callback(shared_returns)

    # Train
    alg_module = importlib.import_module(cfg['algorithm']['module'])
    alg_class = getattr(alg_module, cfg['algorithm']['class'])
    logging.debug(alg_class)

    alg_class(env, policy_fn,
              max_timesteps       = cfg['algorithm']['parameters']['max_timesteps']
              timesteps_per_batch = cfg['algorithm']['parameters']['timesteps_per_batch']
              max_kl              = cfg['algorithm']['parameters']['max_kl']
              cg_iters            = cfg['algorithm']['parameters']['cg_iters']
              cg_damping          = cfg['algorithm']['parameters']['cg_damping']
              vf_iters            = cfg['algorithm']['parameters']['vf_iters']
              vf_stepsize         = cfg['algorithm']['parameters']['vf_stepsize']
              gamma               = cfg['algorithm']['parameters']['gamma']
              lam                 = cfg['algorithm']['parameters']['lam']
              callback            = kindred_callback
              )

    # Safely terminate plotter process
    log_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()
    """

def log_function(env, batch_size, shared_returns, log_running, log_dir): 
    """ Process for logging all data generated during runtime.
    Args:
	    env: An instance of ReacherEnv
        batch_size (int): An int representing timesteps_per_batch provided to the TRPO learn function
        shared_returns (dict): A manager dictionary object containing 'episodic returns' and 'episodic lengths'
        log_running (fn): A multiprocessing Value object containing 0/1 - a flag to allow logging while process is running
    """

    old_size = len(shared_returns['episodic_returns']) 
    time.sleep(5.0)
    rets = []

    # Create logs directory
    if not os.path.exists(log_dir): os.makedirs(log_dir)

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

def load_cfg(yaml_filepath):
    """ Load a YAML configuration file.

    Args:
        yaml_filepath (str): -

    Returns:
        cfg (dict): -
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """ Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): -
        cfg (dict): -

    Returns:
        cfg (dict): -
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.warning("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Benchmarking DRL on Real-World Robots',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.filename)