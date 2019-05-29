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
parser.add_argument("--output-filename",
                    help="filename of plot",
                    metavar="FILENAME",
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
    """Reads all CSV files in a directory
    
    Arguments:
        path {str} -- Path to directory containing the CSV files
    
    Returns:
        numpy array -- All data read from the CSV files
    """
    data = []
    files = os.listdir(path)
    print(files)

    if len(files) == 0:
        logging.error("No files found in dir: {}".format(path))
        return

    for f in files:
        d = np.genfromtxt(os.path.join(path,f), delimiter=',', skip_header=1)
        if len(d) == 36: data.append(d)
    
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
        logging.info("Processing configuration: {0}".format(c))

        # Create path
        c_path = os.path.join(path, c)

        # Obtain data from log files in dir.
        d = read_csv_files(c_path)
        if d is None: continue
        
        data.append(d)

    # Move axis
    data = np.array(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = np.moveaxis(data[i][j],0,1)   

    # Retrieve timestemps for configuration 0, run 0
    timesteps = data[0][0][1]

    # Retrieve rewards
    rewards = []
    for i in range(len(data)):
        rewards.append([r[2] for r in data[i]])
    
    rewards = np.array(rewards)
    for i in range(len(rewards)):
        rewards[i] = np.moveaxis(rewards[i],0,1)  

    # Compute mean reward and its standard deviation
    rewards_mean = []
    rewards_std_dev = []

    #rewards = np.moveaxis(rewards, 1, 2)
    for r in rewards:
        conf_reward_means = []
        conf_reward_std_dev = []
        for v in r:
            conf_reward_means.append(np.mean(v))
            conf_reward_std_dev.append(np.std(v))
            print(v)
        rewards_mean.append(conf_reward_means)
        rewards_std_dev.append(conf_reward_std_dev)
        print("std:dev")
        print(conf_reward_std_dev)
    
    # orange: ff7f0e
    # blue: 1f77b4
    # green: 2ca02c
    # red: d62728
    # violet: ad3fbf
    # magenta: 00aec4
    # purple: 5e2ab5
    # light-green: c1cd14
    colors=["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]

    for i in range(len(rewards_mean)):
        plt.plot(timesteps, rewards_mean[i], linewidth=2.0, label='Configuration {}'.format(i+1),color=colors[i])
        plt.fill_between(timesteps,
                     (np.array(rewards_mean[i]) - np.array(rewards_std_dev[i])),
                     (np.array(rewards_mean[i]) + np.array(rewards_std_dev[i])),
                     color=colors[i],
                     linewidth=0.0,
                     alpha=0.3)

    plt.xticks(np.arange(3e4, 17e4,step=3e4),('30K\n36 min','60K\n72 min','90K\n108 min','120K\n 144 min','150K\n180 min'))
    plt.xlabel('Timesteps',fontweight='bold', labelpad=0)
    plt.ylabel('Average Returns',fontweight='bold')
    plt.title('TRPO Top 5 Configurations',fontweight='bold')

    plt.figure(1)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.output_path,args.output_filename))