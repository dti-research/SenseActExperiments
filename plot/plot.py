# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot results of experiments."""

import os
import io
import sys
import csv
import math
import glob
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Setup argparser
parser = argparse.ArgumentParser(description='Plots results of RL experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",
                    dest="path",
                    help="path to experiment folder containing log files",
                    metavar="PATH",
                    required=True)
parser.add_argument("--output-path",
                    dest="output_path",
                    help="path to output folder",
                    metavar="PATH",
                    required=True)
args = parser.parse_args()

# Setup logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def read_csv_files(path):
    """ Read all csv files in dir

    """
    data = []
    files = os.listdir(path)

    if len(files) == 0:
        logging.error("No files found in dir: {}".format(path))
        return

    for f in files:
        d = np.genfromtxt(os.path.join(path,f), delimiter=',', skip_header=1)
        if len(d) >= 36: data.append(d)
    
    return data

if __name__ == '__main__':
    # Generate output dir
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    
    path = args.path
    configurations = os.listdir(path)
    configurations.sort()
    configurations = [c for c in configurations if glob.glob(os.path.join(path,c) + "/*.csv")]
    
    # Empty placeholder for all configurations
    data = []

    for c in configurations:
        if c == 'same_seed': continue # Ignore same_seed folder
        
        # Create path
        c_path = os.path.join(path, c)

        # Obtain data from log files in dir.
        d = read_csv_files(c_path)
        if d is None: continue
        
        data.append(d)

    # Move axis
    data = np.moveaxis(data, 2, 3)

    # Retrieve timestemps for configuration 0, run 0
    timesteps = data[0][0][1]

    # Retrieve rewards
    rewards = []
    for i in range(len(data)):
        rewards.append([r[2] for r in data[i]])
    
    # Compute mean reward and its standard deviation
    rewards_mean = []
    rewards_std_dev = []

    rewards = np.moveaxis(rewards, 1, 2)
    for r in rewards:
        conf_reward_means = []
        conf_reward_std_dev = []
        for v in r:
            conf_reward_means.append(np.mean(v))
            conf_reward_std_dev.append(np.std(v))
        rewards_mean.append(conf_reward_means)
        rewards_std_dev.append(conf_reward_std_dev)
    
    # green: 2ca02c
    # blue: 1f77b4
    # orange: ff7f0e
    # red: d62728

    plt.plot(timesteps, rewards_mean[0], linewidth=3.0, label='Configuration 1',color='#ff7f0e')
    plt.plot(timesteps, rewards_mean[1], linewidth=3.0, label='Configuration 2',color='#2ca02c')

    plt.fill_between(timesteps,
                     (np.array(rewards_mean[0]) - np.array(rewards_std_dev[0])),
                     (np.array(rewards_mean[0]) + np.array(rewards_std_dev[0])),
                     color='#ff7f0e',
                     linewidth=0.0,
                     alpha=0.3)
    plt.fill_between(timesteps,
                     (np.array(rewards_mean[1]) - np.array(rewards_std_dev[1])),
                     (np.array(rewards_mean[1]) + np.array(rewards_std_dev[1])),
                     color='#2ca02c',
                     linewidth=0.0,
                     alpha=0.3)

    plt.xticks(np.arange(3e4, 17e4,step=3e4),('30K\n36 min','60K\n72 min','90K\n108 min','120K\n 144 min','150K\n180 min'))
    plt.xlabel('Timesteps',fontweight='bold', labelpad=0)
    plt.ylabel('Average Returns',fontweight='bold')
    plt.title('TRPO Top 5 Configurations',fontweight='bold')

    plt.figure(1)
    plt.legend(loc='lower right')
    plt.savefig('trpo.svg')