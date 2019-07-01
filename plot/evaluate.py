import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bas
import bootstrapped.stats_functions as bs_stats
import statsmodels.stats.api as sms

from fitter import Fitter

# Specify where to find log files
path = "../code/artifacts/logs/trpo/0/"
name = "TRPO Configuration 1" # Algorithm and configuration number used for plot titles
filename = "trpo1" # Used for saving plots
files = os.listdir(path) 

multidim = []

# Obtain data from logs
for file in files:
    data = np.genfromtxt(os.path.join(path,file), delimiter=',', dtype=(float, float), skip_header=1)
    multidim.append(data)

# Retrieve timesteps
time = []
for entry in multidim[0]:
    if math.isnan(entry[2]): # If there is no reward value at logging instance
        continue
    else:
        time.append(entry[1])

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

"""
# Perform bootstrapping on data
R = 10000 # Number of resamples
means = []
n = len(r) # Size of resamples

for i in range(R):
    sampled_data = np.random.choice(r,n, replace=True)
    
#plt.hist(sampled_data, bins=100)
#plt.show()
	
    mean = np.average(sampled_data)
    means.append(mean)

plt.hist(means, bins=50)
plt.title(('Empirical Distribution of ' + name),fontweight='bold') 

#plt.figure(1)
#plt.savefig('plots/' + filename + '_dist.pdf')

# TODO fit empirical distribution to the most suited one to determine proper statistic


"""

# Estimate mean using bootstrapping
sim = bas.bootstrap(np.array(avg_rews), stat_func=bs_stats.mean)
print("95 percent CI: \n mean: %.2f (lower: %.2f, upper: %.2f)\n\n" % (sim.value, sim.lower_bound, sim.upper_bound))

# Estimate distribution using bootstrapping
dist = bas.bootstrap(np.array(avg_rews), stat_func=bs_stats.mean, return_distribution=True)

#, 

# Distributions to consider when fitting
distributions=['alpha', 'anglit', 'argus', 'betaprime', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'exponweib','exponnorm', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'gennorm', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'loglaplace', 'maxwell', 'moyal', 'norm', 'pearson3', 'powerlognorm', 'powernorm','reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 'skewnorm', 't', 'trapz', 'triang', 'truncnorm', 'tukeylambda', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max']

f = Fitter(dist, distributions=distributions)
f.fit()
f.summary()
print(f.get_best())



# Retrieve rewards at each timestep 
rew_cols = []
for i in range(len(multidim[0])):
    rew_rows = []
    for file in multidim:
        rew_rows.append(file[i][2])
    rew_cols.append(rew_rows)

# Calculate means and standard errors for each run
rew_std_devs = []
rew_means = []
rew_std_errs = []

#Calculate mean and standard deviation of rewards
for rewards in rew_cols:
    rew_means.append(np.mean(rewards))
    rew_std_errs.append(np.std(rewards, ddof=1)/math.sqrt(len(rewards)))

# Values that may be useful for calculating statistics
print('Average rewards: \n {} \n\n'.format(avg_rews))
print('Standard errors: \n {} \n\n'.format(rew_std_errs))
print('Sample mean: {} \n\n'.format(np.mean(avg_rews)))
print('Sample standard deviation: {} \n\n'.format(np.std(avg_rews)))








