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

    if len(files) == 0:
        logging.error("No files found in dir: {}".format(path))
        return

    for f in files:
        # Check that there is data in the file
        if os.stat(os.path.join(path,f)).st_size == 0: continue
        
        # Read in the data
        d = np.genfromtxt(os.path.join(path,f), delimiter=',', skip_header=1)
        data.append(d)
    
    return data

if __name__ == '__main__':
    # Generate output dir
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    
    path = args.path
    data = read_csv_files(path)

    # Move axis
    data = np.moveaxis(data, 1,2)

    # Retrieve timestemps
    timesteps = data[0][1]

    # Retrieve rewards
    rewards = [r[2] for r in data]

    # Plot
    colors=["#ff7f0e", # orange
            "#1f77b4", # blue
            "#2ca02c", # green
            "#d62728", # red
            "#ad3fbf", # violet
            "#00aec4", # magenta
            "#5e2ab5", # purple
            "#c1cd14", # light-green
            "#ff7f0e", # orange
            "#1f77b4", # blue
            "#2ca02c"] # green

    for i in range(len(rewards)):
        plt.plot(timesteps, rewards[i], linewidth=1.0, label='', color=colors[i]) # label='Run {}'.format(i+1),

    plt.xticks(np.arange(3e4, 17e4,step=3e4),('30K\n36 min','60K\n72 min','90K\n108 min','120K\n 144 min','150K\n180 min'))
    plt.xlabel('Timesteps',fontweight='bold', labelpad=0)
    plt.ylabel('Average Returns',fontweight='bold')
    plt.ylim((-100,300))
    #plt.title('TRPO Configuration 1 - Same Seed\nSimulated Robot',fontweight='bold')
    plt.title('TRPO Configuration 1 - Same Seed\nReal-World Robot',fontweight='bold')
    #plt.title('PPO Configuration 4 - Failed Run (9/10)', fontweight='bold')

    plt.figure(1)
    plt.savefig(os.path.join(args.output_path,args.output_filename))
