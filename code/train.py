# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an experiment."""

import os
import io
import logging
import sys
import ruamel.yaml as yaml
import imp
import importlib
import argparse

# Setup argparser
parser = argparse.ArgumentParser(description='Benchmarking DRL on Real-World Robots',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file",
                    dest="filename",
                    help="experiment definition file",
                    metavar="FILE",
                    required=True)
args = parser.parse_args()

# Setup logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


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
    #cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
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

if __name__ == '__main__':
    yaml_filepath = args.filename

    # Load yaml configuration file and print the configuration
    cfg = load_cfg(yaml_filepath)
    logging.debug(yaml.dump(cfg, default_flow_style=False))

    # Load artifact path
    artifact_path = cfg['train']['artifact_path']
    logging.debug(artifact_path)
    if not os.path.exists(artifact_path): os.makedirs(artifact_path)

    # Load train script
    m = cfg['algorithm']['module']
    t = imp.load_module(m, *imp.find_module(m))
    train = getattr(t, cfg['algorithm']['class'])

    # Train the agent
    train(cfg)
