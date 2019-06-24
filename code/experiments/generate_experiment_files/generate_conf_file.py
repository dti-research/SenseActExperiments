# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import time
import copy
import os
import io
import sys
import datetime
import argparse
import logging
import ruamel.yaml as yaml

# Setup logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

parser = argparse.ArgumentParser(description='Generate experiment files')
parser.add_argument('--filename', type=str, default=None,
                    help='path to read example experiment file')
parser.add_argument('--conf', type=int, default=None,
                    help='Configuration number (default: None)')                    
parser.add_argument('--seed_low', type=int, default=None,
                    help='Lower bound of seed interval (default: None)')
parser.add_argument('--seed_high', type=int, default=None,
                    help='Higher bound of seed interval (default: None)')
parser.add_argument('--hid_size', type=int, default=None,
                    help='size of hidden layers (default: None)')
parser.add_argument('--num_hid_layers', type=int, default=None,
                    help='number of hidden layers (default: None)')
parser.add_argument('--batch_size', type=int, default=None,
                    help='input batch size for training (default: None)')
parser.add_argument('--timesteps_per_batch', type=int, default=None,
                    help='input batch size for training (default: None)')
parser.add_argument('--vf_stepsize', type=float, default=None,
                    help='stepsize (default:None)')
parser.add_argument('--max_kl', type=float, default=None,
                    help='max_kl (default: None)')
parser.add_argument('--gamma', type=float, default=None,
                    help='gamma (default: None)')
parser.add_argument('--lamda', type=float, default=None,
                    help='lamda (default: None)')
parser.add_argument('--opt_batch_size', type=int, default=None,
                    help='optimizer batch size for training (default: None)')
parser.add_argument('--opt_step_size', type=float, default=None,
                    help='optimizer step-size (default: None)')
parser.add_argument('--output_filename', type=str, default=None,
                    help='Experiment output filename')
parser.add_argument('--output_dir', type=str, default=None,
                    help='path to put experiment files')
parser.add_argument('--log_dir', type=str, default=None,
                    help='path to put log files')
args = parser.parse_args()

def set_trpo_hyperparameters(cfg):
    """ Sets all TRPO specific hyperparameters.

    Args:
        cfg (dict) : Loaded YAML configuration file
    
    Returns
        cfg (dict) : Updated YAML dict
    """
    cfg['algorithm']['hyperparameters']['timesteps_per_batch']  = args.timesteps_per_batch
    cfg['algorithm']['hyperparameters']['max_kl']               = args.max_kl
    cfg['algorithm']['hyperparameters']['vf_stepsize']          = args.vf_stepsize    
    cfg['algorithm']['hyperparameters']['gamma']                = args.gamma
    cfg['algorithm']['hyperparameters']['lam']                  = args.lamda
    cfg['algorithm']['hyperparameters']['hid_size']             = args.hid_size    
    cfg['algorithm']['hyperparameters']['num_hid_layers']       = args.num_hid_layers        

    return cfg

def set_ppo_hyperparameters(cfg):
    """ Sets all PPO specific hyperparameters.

    Args:
        cfg (dict) : Loaded YAML configuration file
    
    Returns
        cfg (dict) : Updated YAML dict
    """

    cfg['algorithm']['hyperparameters']['batch_size']           = args.batch_size
    cfg['algorithm']['hyperparameters']['optim_batchsize']      = args.opt_batch_size
    cfg['algorithm']['hyperparameters']['optim_stepsize']       = args.opt_step_size
    cfg['algorithm']['hyperparameters']['gamma']                = args.gamma
    cfg['algorithm']['hyperparameters']['lam']                  = args.lamda
    cfg['algorithm']['hyperparameters']['hid_size']             = args.hid_size    
    cfg['algorithm']['hyperparameters']['num_hid_layers']       = args.num_hid_layers        


    return cfg

def load_cfg(yaml_filepath):
    """ Load a YAML configuration file.

    Args:
        yaml_filepath (str): -

    Returns:
        cfg (dict): -
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)
    return cfg

if __name__ == '__main__':
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    yaml_filepath = args.filename

    # Load yaml configuration file and print the configuration
    cfg = load_cfg(yaml_filepath)
    #logging.debug(yaml.dump(cfg, default_flow_style=False))

    cfg['global']['seed']['high'] = args.seed_high if not None else None
    cfg['global']['seed']['low'] = args.seed_low if not None else None

    cfg['train']['artifact_path'] = args.log_dir

    # Set TRPO specific hyper-parameters
    cfg = set_trpo_hyperparameters(cfg)

    # Set PPO specific hyper-parameters
    #cfg = set_ppo_hyperparameters(cfg)

    # Save configuration file
    new_conf_file = os.path.join(args.output_dir, args.output_filename)
    logging.debug('Saving configuration file as: {}'.format(new_conf_file))
    with open(new_conf_file, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
