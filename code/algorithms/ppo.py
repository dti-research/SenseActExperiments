# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import copy
import numpy as np
import os
import io
import logging
import imp
import importlib

from multiprocessing import Process, Value, Manager

from helper import create_callback

l = imp.find_module('utils/logger')
l = imp.load_module('utils/logger', *l)
log_function = getattr(l, 'log_function')

def train(cfg):
    """ Function to start training and logging processes

    Args:
        cfg (dict): Configuration parameters loaded into dict from yaml file
    """

    artifact_path = cfg['train']['artifact_path']
    
    # Get environment
    m = cfg['environment']['module']
    t = imp.load_module(m, *imp.find_module(m))
    get_env = getattr(t, cfg['environment']['class'])
    env, policy_fn = get_env(cfg)

    # Create and start logging process
    log_running = Value('i', 1)
    # Manager to share data between log process and main process
    shared_returns = Manager().dict({'write_lock': False,
                                     'episodic_returns': [],
                                     'episodic_lengths': [], }) 
    
    # Spawn logging process
    pp = Process(target=log_function,
                 args=(env,
                       cfg['algorithm']['hyperparameters']['batch_size'],
                       shared_returns,
                       log_running,
                       artifact_path
                       )
                )
    pp.start()

    # Create callback function for logging data from learn
    kindred_callback = create_callback(shared_returns)

    # Train
    m = importlib.import_module(cfg['algorithm']['codebase']['module'])
    learn = getattr(m, cfg['algorithm']['codebase']['class'])
    logging.debug(learn)
    logging.debug(type(cfg['algorithm']['hyperparameters']['adam_epsilon']))

    learn(env, policy_fn,
          max_timesteps            = cfg['algorithm']['hyperparameters']['max_timesteps'],
          timesteps_per_actorbatch = cfg['algorithm']['hyperparameters']['batch_size'],
          clip_param               = cfg['algorithm']['hyperparameters']['clip_param'],
          entcoeff                 = cfg['algorithm']['hyperparameters']['entcoeff'], 
          optim_batchsize          = cfg['algorithm']['hyperparameters']['optim_batchsize'],
          optim_epochs             = cfg['algorithm']['hyperparameters']['optim_epochs'],
          optim_stepsize           = cfg['algorithm']['hyperparameters']['optim_stepsize'],
          adam_epsilon             = cfg['algorithm']['hyperparameters']['adam_epsilon'],
          schedule                 = cfg['algorithm']['hyperparameters']['schedule'],
          gamma                    = cfg['algorithm']['hyperparameters']['gamma'],
          lam                      = cfg['algorithm']['hyperparameters']['lam'],
          callback                 = kindred_callback
          )
    
    # Safely terminate plotter process
    log_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()