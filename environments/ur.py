# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import imp
import importlib

import numpy as np
from senseact.utils import tf_set_seeds

def get_env(cfg):
    # Use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Load the RL Environment
    env_module = importlib.import_module(cfg['environment']['codebase']['module'])
    env_class = getattr(env_module, cfg['environment']['codebase']['class'])
    logging.debug(env_class)

    # Load setup file
    s = imp.find_module(cfg['environment']['setup']['module'])
    s = imp.load_module(cfg['environment']['setup']['module'], *s)
    setup = getattr(s, cfg['environment']['setup']['class'])

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
    
    return env, policy_fn
