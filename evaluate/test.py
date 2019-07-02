from scipy import stats
data = stats.gamma.rvs(2, loc=1.5, scale=2, size=100000)

from fitter import Fitter
f = Fitter(data, distributions=['gamma', 'rayleigh', 'uniform'])
f.fit()
print("here")
print(f.summary())
print("here")

f.fitted_pdf['gamma']

print("here")