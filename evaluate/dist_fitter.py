# Copyright (c) 2019, Danish Technological Institute.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bas
import bootstrapped.stats_functions as bs_stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(20)


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

def load_all_distributions():
    distributions = []
    for this in dir(scipy.stats):
        if "fit" in eval("dir(scipy.stats." + this +")"):
            distributions.append(this)
    return distributions



# Specify where to find log files
path = "../code/artifacts/logs/trpo/1/"
name = "TRPO Configuration 1" # Algorithm and configuration number used for plot titles
filename = "trpo1" # Used for saving plots
files = os.listdir(path) 

multidim = []

# Obtain data from logs
for file in files:
    data = np.genfromtxt(os.path.join(path,file), delimiter=',', dtype=(float, float), skip_header=1)
    multidim.append(data)

# Calculate average reward
rews = []
for file in multidim:
    rows = []
    for i in range(1, len(multidim[0])):
        rows.append(file[i][2])
    rews.append(rows)

avg_rews = []
for r in rews:
    avg_rews.append(np.mean(r))

# Estimate distribution using bootstrapping
d = bas.bootstrap(np.array(avg_rews), stat_func=bs_stats.mean, return_distribution=True)

# plot normed histogram
plt.title(('Empirical Distribution of ' + name + '\nwith Fitted Theoretical Distributions'),fontweight='bold') 
plt.hist(d, bins=int(math.sqrt(len(d))), density=True, alpha=0.4, edgecolor='k')

## All distributions (100)
#distributions = load_all_distributions()

## All converging distributions (52)
##  - Excluded from fitting (48): 'anglit', 'arcsine', 'bradford', 'chi', 'expon', 'exponpow', 'exponweib', 'foldnorm', 'frechet_r', 'genexpon', 'gengamma', 'genhalflogistic', 'genpareto', 'gilbrat', 'gompertz', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'kappa3', 'kappa4', 'ksone', 'levy_l', 'levy', 'levy_stable', 'lomax', 'maxwell', 'nakagami', 'ncx2', 'pareto', 'pearson3', 'powerlaw', 'rayleigh', 'reciprocal', 'rice', 'rdist', 'recipinvgauss', 'semicircular', 'rv_continuous', 'rv_histogram', 'trapz', 'truncexpon', 'truncnorm', 'uniform', 'vonmises', 'wald', 'weibull_min', 'wrapcauchy'
#distributions = ['alpha', 'argus', 'beta', 'betaprime',  'burr', 'burr12', 'cauchy', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'exponnorm', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'frechet_l', 'gamma', 'gausshyper', 'genextreme', 'genlogistic', 'gennorm', 'gumbel_l', 'gumbel_r', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kstwobign', 'laplace', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'mielke', 'moyal', 'ncf', 'nct', 'norm', 'norminvgauss', 'powerlognorm', 'powernorm', 'skewnorm', 't', 'triang', 'tukeylambda', 'vonmises_line', 'weibull_max']

# Top-12 Distributions
distributions = ['beta', 'crystalball', 'exponnorm', 'f', 'gennorm', 'johnsonsb', 'johnsonsu', 'loggamma', 'norm', 'powernorm', 'skewnorm', 't']

# find minimum and maximum of xticks, so we know
# where we should compute theoretical distribution
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(d))

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
        if p==0.0:
            print("{}\t{}\t\t\t{}: {}".format(statistic, p, distribution, params))
        else:
            print("{}\t{}\t{}: {}".format(statistic, p, distribution, params))
        
        # Save values for later
        dist_params[distribution] = params
        dist_names.append(distribution)
        ks_test_p.append(p)
        ks_test_statistics.append(statistic)

        # Plot PDF in histogram
        if p == 0.0: continue
        plt.plot(lnspc, pdf, label=distribution)
        
    except Exception as e:
        print(e)

# Print unsorted lists
print(ks_test_p)
print(dist_names)

# Sort the lists together (keeping them in sync)
l1, l2 = (list(t) for t in zip(*sorted(zip(ks_test_p, dist_names))))
print(l1)
print(l2)

# Look up parameters using the dist name, now sorted after p-value
print(dist_params[l2[0]])

# Show the histogram with the fitted PDFs
if len(distributions) < 15:
    plt.legend()
plt.show()
