import os
import csv
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import math
import statsmodels.stats.api as sms
import pandas as pd
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

f = Fitter(means)
f.fit()
f.summary()



# TODO fit empirical distribution to the most suited one to determine proper statistic

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
print('\nAverage rewards: \n {} \n\n'.format(avg_rews))
print('Standard errors: \n {} \n\n'.format(rew_std_errs))
print('Sample mean: {} \n\n'.format(np.mean(avg_rews)))
print('Sample standard deviation: {} \n\n'.format(np.std(avg_rews)))








