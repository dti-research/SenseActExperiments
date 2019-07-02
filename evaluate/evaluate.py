# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluate the results of concucted trials.
    Step 1: Read in the average rewards and
            timesteps for the specified configuration
    Step 2: 
"""

import os
import io
import sys
import csv
import math
import glob
import logging
import argparse

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as st
import bootstrapped.bootstrap as bas
import bootstrapped.stats_functions as bs_stats


# Setup argparser
parser = argparse.ArgumentParser(description='Evaluates results of RL experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",
                    dest="path",
                    help="path to experiment folder containing log files",
                    metavar="PATH",
                    required=True)
parser.add_argument("--algorithm",
                    help="name of algorithm",
                    metavar="NAME",
                    required=True)
parser.add_argument("--configuration",
                    help="number configuration to evaluate",
                    metavar="NUMBER",
                    type=int,
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

def bootstrap(rewards, n_resamples=10000, seed=None):
    if seed is not None: np.random.seed(seed)

    avg_rewards = []
    for r in rewards:
        avg_rewards.append(np.mean(r))

    means = []
    for i in range(n_resamples):
        sampled_data = np.random.choice(avg_rewards, len(rewards), replace=True)        
        means.append(np.mean(sampled_data))

    return means

def compute_mean_ci(data, confidence=0.95):
    #a = 1.0 * np.array(data)
    n = len(data)
    m, se = np.mean(data), st.sem(data)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == '__main__':
    # Generate output dir
    log_dir = args.output_path
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    # Read in data
    path = args.path
    data = read_csv_files(path)

    # Open file to write to
    csv_file_path = os.path.join(log_dir, args.algorithm + '_conf_' + str(args.configuration) + '.csv')
    f = open(csv_file_path, 'w')

    f.write('Mean, Lower Bound, Upper Bound, BS Mean, BS Lower Bound, BS Upper Bound\n')
    

    # Retrieve rewards
    rewards = []
    for d in data:
        rows = []
        for i in range(len(data[0])):
            r = d[i][2]
            if math.isnan(r):
                continue
            rows.append(r)
        rewards.append(rows)
    
    # Bootstrap data
    resampled_data = bootstrap(rewards, n_resamples=10000, seed=42)

    # Compute mean and 95% CI for the resampled data
    mu, lower, upper = compute_mean_ci(resampled_data, confidence=0.95)
    print(" mean: {} (lower: {}, upper: {})".format(mu, lower, upper))

    print(st.t.interval(0.05, len(resampled_data)-1, loc=np.mean(resampled_data), scale=st.sem(resampled_data)))

    # Estimate mean with 95% CI using bootstrapping
    avg_rews = []
    for r in rewards:
        avg_rews.append(np.mean(r))
    sim = bas.bootstrap(np.array(avg_rews), stat_func=bs_stats.mean)   
    print(sim.value)

    f.write(str(mu) + ',' +
            str(lower) + ',' +
            str(upper) + ',' +
            str(sim.value) + ',' +
            str(sim.lower_bound) + ',' +
            str(sim.upper_bound) + '\n')

    f.close()

    k2, p = st.normaltest(resampled_data)
    alpha = 0.05
    print("p = {:g}".format(p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")

    (mu, sigma) = st.norm.fit(resampled_data)

    # Create histogram plot
    bins=50
    plt.hist(resampled_data, bins=bins, density=True, alpha=0.4, edgecolor='k')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = st.norm.pdf(x, mu, sigma)
    
    plt.plot(x, pdf, 'r', linewidth=2)
    plt.title(('Empirical Distribution of ' + args.algorithm.upper() + ' Configuration ' + str(args.configuration+1)),fontweight='bold')
    #plt.xlabel('Average Returns', labelpad=0)
    #plt.ylabel('$n$-samples')
    #plt.show()
    #plt.ylim(0,700)
    #plt.xlim(80,160)
    plt.savefig(os.path.join(log_dir, args.algorithm.lower() + '_conf_' + str(args.configuration) + '_dist.pdf'))
