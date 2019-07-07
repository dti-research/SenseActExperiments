# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bas
import bootstrapped.stats_functions as bs_stats
import warnings
warnings.filterwarnings('ignore')

# Setup argparser
parser = argparse.ArgumentParser(description='Fits theoretical distributions to empirical ones',
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
parser.add_argument("--value",
                    help="reported value to test against",
                    metavar="AVG_RETURN",
                    type=float,
                    required=True)
parser.add_argument("--seed",
                    help="seed for the random number generator",
                    metavar="SEED",
                    type=int,
                    required=True)
parser.add_argument("--output-path",
                    dest="output_path",
                    help="path to output folder",
                    metavar="PATH",
                    required=True)
args = parser.parse_args()

# Seed for reproducibility
# Seed 20 result: powernorms
np.random.seed(args.seed)

# Keep for bookkeeping!
#def compute_kl_div(p, q):
#    if math.isnan(p) or math.isnan(q):
#        return nan
#    elif p > 0 and q > 0:
#        #return p * math.log(p / q) - p + q
#        return p * math.log(p / q)
#    elif x == 0 and y >= 0:
#        return y
#    else:
#        return inf
#
#def compute_rel_entr(p, q):
#    if math.isnan(p) or math.isnan(q):
#        return nan
#    elif p > 0 and q > 0:
#        return p * math.log(p / q)
#    elif p == 0 and q >= 0:
#        return 0
#    else:
#        return inf

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

def load_all_distributions():
    """[summary]
    
    Returns:
        [type] -- [description]
    """
    distributions = []
    for this in dir(scipy.stats):
        if "fit" in eval("dir(scipy.stats." + this +")"):
            distributions.append(this)
    return distributions

def get_xticks(plt, data):
    """[summary]
    
    Arguments:
        plt {[type]} -- [description]
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(data))
    return lnspc

def fit_distributions(data, distributions, lnspc, output_dir, output_postfix):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        distributions {[type]} -- [description]
        lnspc {[type]} -- [description]
        output_dir {[type]} -- [description]
        output_postfix {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Open file to write to
    #csv_file_path = os.path.join(log_dir, args.algorithm + '_conf_' + str(args.configuration) + '.csv')
    csv_file_path = os.path.join(output_dir, args.algorithm + '_conf_' + str(args.configuration) + output_postfix + '_52.csv')
    f = open(csv_file_path, 'w')
    
    # Write header
    f.write('Distribution; Statistics; p-value; Parameters\n')

    # Placeholders
    dist_params = {}
    dist_names = []
    ks_test_p = []
    ks_test_statistics = []

    print("-----------------------------------------------------------------------------------------------------------------------------")
    print("Statistics \t\tp-value \t\tDistribution")
    print("-----------------------------------------------------------------------------------------------------------------------------")
    for distribution in distributions:
        try:
            # Get callable
            dist = getattr(scipy.stats, distribution)

            # Fit theoretical distribution to empirical one
            params = dist.fit(d)

            # Extract the PDF using the generated parameters
            pdf = dist.pdf(lnspc, *params)

            # Determine goodness of fit by Kolmogorov-Smirnov test
            (statistic, p) = scipy.stats.kstest(d, distribution, params)
            #if p==0.0:
            #    print("{}\t{}\t\t\t{}: {}".format(statistic, p, distribution, params))
            #else:
            #    print("{}\t{}\t{}: {}".format(statistic, p, distribution, params))

            # Print LaTeX table
            latex_params = tuple([float("{:.2f}".format(n)) for n in params])
            dist_name = distribution.replace("_", "\\_")
            print("{} & {:.4f} & {:.4f} & {} \\\\".format(distribution, statistic, p, latex_params))
            
            # Save values for later
            dist_params[distribution] = params
            dist_names.append(distribution)
            ks_test_p.append(p)
            ks_test_statistics.append(statistic)

            # Save values to disk
            f.write(str(distribution) + ';' +
                    str(statistic) + ';' +
                    str(p) + ';' +
                    str(params) + '\n')

            # Plot PDF in histogram
            if p == 0.0: continue
            plt.plot(lnspc, pdf, label=distribution)
            
        except Exception as e:
            print(e)
    
    # Close file
    f.close()

    # Save the histogram with the fitted PDFs
    if len(distributions) < 15:
        plt.legend()
    #plt.savefig(os.path.join(log_dir, args.algorithm.lower() + '_conf_' + str(args.configuration) + '_fitted dists.pdf'))
    plt.savefig(os.path.join(output_dir, args.algorithm.lower() + '_conf_' + str(args.configuration) + output_postfix + '.pdf'))

    return dist_names, dist_params, ks_test_p, ks_test_statistics

def fit_best_pdf(data, distribution, lnspc, output_dir):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
        distribution {[type]} -- [description]
        lnspc {[type]} -- [description]
        output_dir {[type]} -- [description]
    """
    # Get callable
    dist = getattr(scipy.stats, distribution)

    # Fit theoretical distribution to empirical one
    params = dist.fit(data)
    best_pdf = dist.pdf(lnspc, *params)

    # Plot best
    plt.figure(4)
    plt.title(('Empirical Distribution of ' + 
                args.algorithm.upper() + 
            ' Configuration ' + 
                str(args.configuration+1) + 
            '\nwith Best Fitted Theoretical Distribution'),
                fontweight='bold') 
    plt.hist(d, bins=int(math.sqrt(len(d))), density=True, alpha=0.4, edgecolor='k')
    plt.plot(lnspc, best_pdf, label=distribution)
    plt.legend()
    plt.savefig(os.path.join(output_dir, args.algorithm.lower() + '_conf_' + str(args.configuration) + '_best_fit.pdf'))

def compute_probabilities(edf, distributions, lnspc, value):
    # Get values at least as good as reported
    values_at_least_as_good = edf[edf >= value]
    values_at_least_as_good_idx = np.nonzero(edf >= value)[0]

    # Debugging
    #print("Number of samples, better than reported: {}".format(len(values_at_least_as_good_idx)))
    #print("   Values: {}".format(values_at_least_as_good))
    #print("   Indices: {}".format(values_at_least_as_good_idx))
    #print("Value at index {}: {}".format(values_at_least_as_good_idx[0]-1, d[values_at_least_as_good_idx[0]-1]))

    # Normalise to obtain probabilities
    values_at_least_as_good /= edf.sum()
    pv = np.sum(values_at_least_as_good)

    print("-----------------------------------------------------------------------------------------------------------------------------")
    print("Computing probability of obtaining at least as extreme a value as: {}".format(value))
    print("p-value \t\tDistribution")
    print("-----------------------------------------------------------------------------------------------------------------------------")
    for distribution in distributions:
        try:
            # Get callable
            dist = getattr(scipy.stats, distribution)

            # Fit theoretical distribution to empirical one
            params = dist.fit(d)

            # Extract the PDF using the generated parameters
            pdf = dist.pdf(lnspc, *params)

            # Determine goodness of fit by Kolmogorov-Smirnov test
            (statistic, p) = scipy.stats.kstest(d, distribution, params)

            # Compute probability: P(v>threshold|dist=d, data) = Pv * Pd
            pd = p

            if p==0.0:
                print("{:.4f}\t\t{}".format(pv*pd, distribution))
            else:
                print("{:.4f}\t{}".format(pv*pd, distribution))
            
        except Exception as e:
            print(e)

if __name__ == '__main__':
    # Generate output dir
    log_dir = args.output_path
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    # Read in data
    path = args.path
    data = read_csv_files(path)

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

    # Estimate mean
    avg_rews = []
    for r in rewards:
        avg_rews.append(np.mean(r))

    # Estimate distribution using bootstrapping
    d = bas.bootstrap(np.array(avg_rews), stat_func=bs_stats.mean, return_distribution=True)
    d.sort()

    ############################
    ## All distributions (100) #
    ############################
    #distributions = load_all_distributions()
    ## Plot normed histogram with fitted distributions
    #plt.figure(1)
    #plt.title(('Empirical Distribution of ' + args.algorithm.upper() + ' Configuration ' + str(args.configuration+1) + '\nwith Fitted Theoretical Distributions'), fontweight='bold') 
    #plt.hist(d, bins=int(math.sqrt(len(d))), density=True, alpha=0.4, edgecolor='k')
    #dist_names, dist_params, ks_test_p, ks_test_statistics = fit_distributions(data=d,distributions, lnspc, output_dir=log_dir)
    """
    #####################################
    # All converging distributions (52) #
    #####################################
    #  - Excluded from fitting (48): 'anglit', 'arcsine', 'bradford', 'chi', 'expon', 'exponpow', 'exponweib', 'foldnorm', 'frechet_r', 'genexpon', 'gengamma', 'genhalflogistic', 'genpareto', 'gilbrat', 'gompertz', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'kappa3', 'kappa4', 'ksone', 'levy_l', 'levy', 'levy_stable', 'lomax', 'maxwell', 'nakagami', 'ncx2', 'pareto', 'pearson3', 'powerlaw', 'rayleigh', 'reciprocal', 'rice', 'rdist', 'recipinvgauss', 'semicircular', 'rv_continuous', 'rv_histogram', 'trapz', 'truncexpon', 'truncnorm', 'uniform', 'vonmises', 'wald', 'weibull_min', 'wrapcauchy'
    distributions = ['alpha', 'argus', 'beta', 'betaprime',  'burr', 'burr12', 'cauchy', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'exponnorm', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'frechet_l', 'gamma', 'gausshyper', 'genextreme', 'genlogistic', 'gennorm', 'gumbel_l', 'gumbel_r', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kstwobign', 'laplace', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'mielke', 'moyal', 'ncf', 'nct', 'norm', 'norminvgauss', 'powerlognorm', 'powernorm', 'skewnorm', 't', 'triang', 'tukeylambda', 'vonmises_line', 'weibull_max']
    # Plot normed histogram with fitted distributions
    plt.figure(2)
    plt.title(('Empirical Distribution of ' + args.algorithm.upper() + ' Configuration ' + str(args.configuration+1) + '\nwith Fitted Theoretical Distributions'), fontweight='bold') 
    plt.hist(d, bins=int(math.sqrt(len(d))), density=True, alpha=0.4, edgecolor='k')

    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    lnspc = get_xticks(plt, d)

    # Fit distributions to our data
    dist_names, dist_params, ks_test_p, ks_test_statistics = fit_distributions(d,distributions, lnspc, output_dir=log_dir, output_postfix='_52_dists')

    # Sort the lists together (keeping them in sync)
    ks_test_p, dist_names = (list(t) for t in zip(*sorted(zip(ks_test_p, dist_names))))

    print(dist_names)
    
    """
    
    #######################
    # Top-6 Distributions #
    #######################
    #distributions = ['beta', 'crystalball', 'exponnorm', 'f', 'gennorm', 'johnsonsb', 'johnsonsu', 'loggamma', 'norm', 'powernorm', 'skewnorm', 't']
    distributions = ['beta', 'johnsonsb', 'johnsonsu', 'loggamma', 'powernorm', 'skewnorm']
    # Plot normed histogram with fitted distributions
    plt.figure(3)
    plt.title(('Empirical Distribution of ' + args.algorithm.upper() + ' Configuration ' + str(args.configuration+1) + '\nwith Fitted Theoretical Distributions'), fontweight='bold') 
    plt.hist(d, bins=int(math.sqrt(len(d))), density=True, alpha=0.4, edgecolor='k')

    # find minimum and maximum of xticks, so we know
    # where we should compute theoretical distribution
    lnspc = get_xticks(plt, d)

    """
    # Fit distributions to our data
    dist_names, dist_params, ks_test_p, ks_test_statistics = fit_distributions(d,distributions, lnspc, output_dir=log_dir, output_postfix='_6_dists')

    # Sort the lists together (keeping them in sync)
    ks_test_p, dist_names = (list(t) for t in zip(*sorted(zip(ks_test_p, dist_names))))

    fit_best_pdf(d, dist_names[-1], lnspc, log_dir)
    """
    
    # Compute P-values
    compute_probabilities(d, distributions, lnspc, args.value)
    
