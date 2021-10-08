---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Scipy

SciPy is a collection of mathematical algorithms and convenience functions built on the NumPy extension of Python. It adds significant power to the interactive Python session by providing the user with high-level commands and classes for manipulating and visualizing data. With SciPy, an interactive Python session becomes a data-processing and system-prototyping environment. 

The additional benefit of basing SciPy on Python is that this also makes a powerful programming language available for use in developing sophisticated programs and specialized applications. Everything from parallel programming to web and data-base subroutines and classes have been made available to the Python programmer. All of this power is available in addition to the mathematical libraries in SciPy.

```{code-cell} ipython3
#Importing stats from scipy package

from scipy import stats
import numpy as np
```

## Discrete Distributions

+++

### Bernoulli

```{code-cell} ipython3
from scipy.stats import bernoulli
```

```{code-cell} ipython3
## Calculate the first four moments

p = 0.6
mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(bernoulli.ppf(0.99, p))
bernoulli.pmf(x, p)
```

Where bernoulli.ppf is similar to "qbern()" function and bernoulli.pmf is similar to "dbern()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

bernoulli.cdf(x, p)
```

Similar to "pbern()" function in r.

```{code-cell} ipython3
## Generate random numbers

bernoulli.rvs(p, size=10)
```

Similar to "rbern()" function in r.

+++

### Betabinom

```{code-cell} ipython3
from scipy.stats import betabinom
```

```{code-cell} ipython3
## Calculate the first four moments

n, a, b = 7, 1.7, 0.42
mean, var, skew, kurt = betabinom.stats(n, a, b, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(betabinom.ppf(0.99, n, a, b))
betabinom.pmf(x, n, a, b)
```

Where betabinom.ppf is similar to "qbb()" function and betabinom.pmf is similar to "dbb()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

betabinom.cdf(x, n, a, b)
```

Similar to "pbb()" function in r.

```{code-cell} ipython3
## Generate random numbers

betabinom.rvs(n, a, b, size=10)
```

Similar to "rbb()" function in r.

+++

### Binomial

```{code-cell} ipython3
from scipy.stats import binom
```

```{code-cell} ipython3
## Calculate the first four moments

n, p = 7, 0.7
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(binom.ppf(0.99, n, p))
binom.pmf(x, n, p)
```

Where binom.ppf is similar to "qbinom()" function and binom.pmf is similar to "dbinom()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

binom.cdf(x, n, p)
```

Similar to "pbinom()" function in r.

```{code-cell} ipython3
## Generate random numbers

binom.rvs(n, p, size=10)
```

Similar to "rbinom()" function in r.

+++

### Geometric

```{code-cell} ipython3
from scipy.stats import geom
```

```{code-cell} ipython3
## Calculate the first four moments

p = 0.7
mean, var, skew, kurt = geom.stats(p, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(geom.ppf(0.99, p))
geom.pmf(x, p)
```

Where geom.ppf is similar to "qgeom()" function and geom.pmf is similar to "dgeom()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

geom.cdf(x, p)
```

Similar to "pgeom()" function in r.

```{code-cell} ipython3
## Generate random numbers

geom.rvs(p, size=10)
```

Similar to "rgeom()" function in r.

+++

### Hypergeometric

```{code-cell} ipython3
from scipy.stats import hypergeom
```

```{code-cell} ipython3
## Calculate the first four moments

M, n, N = 18, 3, 12
mean, var, skew, kurt = hypergeom.stats(M, n, N, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x =  np.arange(hypergeom.ppf(0.99,M,n,N))
hypergeom.pmf(x,M, n, N)
```

Where hypergeom.ppf is similar to "qhyper()" function and hypergeom.pmf is similar to "dhyper()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

hypergeom.cdf(x, M, n, N)
```

Similar to "phyper()" function in r.

```{code-cell} ipython3
## Generate random numbers

hypergeom.rvs(M, n, N, size=10)
```

Similar to "rhyper()" function in r.

+++

### Negative Binomial

```{code-cell} ipython3
from scipy.stats import nbinom
```

```{code-cell} ipython3
## Calculate the first four moments

n, p = 7, 0.7
mean, var, skew, kurt = nbinom.stats(n, p, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(nbinom.ppf(0.99, n, p))
nbinom.pmf(x, n, p)
```

Where nbinom.ppf is similar to "qnbinom()" function and nbinom.pmf is similar to "dnbinom()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

nbinom.cdf(x, n, p)
```

Similar to "pnbinom()" function in r.

```{code-cell} ipython3
## Generate random numbers

nbinom.rvs(n, p, size=10)
```

Similar to "rnbinom()" function in r.

+++

### Poisson

```{code-cell} ipython3
from scipy.stats import poisson
```

```{code-cell} ipython3
## Calculate the first four moments

mu = 0.7
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(poisson.ppf(0.99, mu))
poisson.pmf(x, mu)
```

Where poisson.ppf is similar to "qpois()" function and poisson.pmf is similar to "dpois()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

poisson.cdf(x, mu)
```

Similar to "ppois()" function in r.

```{code-cell} ipython3
## Generate random numbers

poisson.rvs(mu, size=10)
```

Similar to "rpois()" function in r.

+++

### Uniform

```{code-cell} ipython3
from scipy.stats import randint
```

```{code-cell} ipython3
## Calculate the first four moments

low, high = 5, 25
mean, var, skew, kurt = randint.stats(low, high, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability mass function

x = np.arange(randint.ppf(0.99, low, high))
randint.pmf(x, low, high)
```

Where randint.ppf is similar to "qdunif()" function and randint.pmf is similar to "ddunif()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

randint.cdf(x, low, high)
```

Similar to "pdunif()" function in r.

```{code-cell} ipython3
## Generate random numbers

randint.rvs(low, high, size=10)
```

Similar to "rdunif()" function in r.

+++

Few more distributions that are also supported in this package is:
- Boltzmann (truncated Planck) Distribution
- Planck (discrete exponential) Distribution
- Fisher’s Noncentral Hypergeometric Distribution
- Wallenius’ Noncentral Hypergeometric Distribution
- Negative Hypergeometric Distribution
- Zipf (Zeta) Distribution
- Zipfian Distribution
- Logarithmic (Log-Series, Series) Distribution
- Discrete Laplacian Distribution
- Yule-Simon Distribution

+++

## Continuous Distributions

+++

### Beta

```{code-cell} ipython3
from scipy.stats import beta
```

```{code-cell} ipython3
## Calculate the first four moments

a, b = 1.76, 0.35
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(beta.ppf(0.99, a, b), 100)
beta.pdf(x, a, b)
```

Where beta.ppf is similar to "qbeta()" function and beta.pmf is similar to "dbeta()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

beta.ppf([0.001, 0.5, 0.999], a, b)
```

Similar to "pbeta()" function in r.

```{code-cell} ipython3
## Generate random numbers

beta.rvs(a, b, size=10)
```

Similar to "rbeta()" function in r.

+++

### Cauchy

```{code-cell} ipython3
from scipy.stats import cauchy
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(cauchy.ppf(0.99), 100)
cauchy.pdf(x)
```

Where cauchy.ppf is similar to "qcauchy()" function and cauchy.pmf is similar to "dcauchy()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

cauchy.ppf([0.01, 0.5, 0.999])
```

Similar to "pcauchy()" function in r.

```{code-cell} ipython3
## Generate random numbers

cauchy.rvs(size=10)
```

Similar to "rcauchy()" function in r.

+++

### Chi square

```{code-cell} ipython3
from scipy.stats import chi2
```

```{code-cell} ipython3
## Calculate the first four moments

df = 30
mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(chi2.ppf(0.99, df), 100)
chi2.pdf(x, df)
```

Where chi2.ppf is similar to "qchisq()" function and chi2.pmf is similar to "dchisq()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

chi2.ppf([0.001, 0.5, 0.999], df)
```

Similar to "pchisq()" function in r.

```{code-cell} ipython3
## Generate random numbers

chi2.rvs(df, size=10)
```

Similar to "rchisq()" function in r.

+++

### Exponential

```{code-cell} ipython3
from scipy.stats import expon
```

```{code-cell} ipython3
## Calculate the first four moments

loc, scale = 16, 3
mean, var, skew, kurt = expon.stats(loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(expon.ppf(0.99), 100)
expon.pdf(x,loc,scale)
```

Where expon.ppf is similar to "qexp()" function and expon.pmf is similar to "dexp()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

expon.ppf([0.001, 0.5, 0.999],loc,scale)
```

Similar to "pexp()" function in r.

```{code-cell} ipython3
## Generate random numbers

expon.rvs(loc,scale,size=10)
```

Similar to "rexp()" function in r.

+++

### Gamma

```{code-cell} ipython3
from scipy.stats import gamma
```

```{code-cell} ipython3
## Calculate the first four moments

a, loc, scale = 5,0, 2
mean, var, skew, kurt = gamma.stats(a,loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(gamma.ppf(0.99, a), 100)
gamma.pdf(x, a,loc,scale)
```

Where gamma.ppf is similar to "qgamma()" function and gamma.pmf is similar to "dgamma()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

gamma.ppf([0.001, 0.5, 0.999], a,loc,scale)
```

Similar to "pgamma()" function in r.

```{code-cell} ipython3
## Generate random numbers

gamma.rvs(scale,a, size=10)
```

Similar to "rgamma()" function in r.

+++

### Laplace

```{code-cell} ipython3
from scipy.stats import laplace
```

```{code-cell} ipython3
## Calculate the first four moments

loc, scale = 16, 3
mean, var, skew, kurt = laplace.stats(loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(laplace.ppf(0.99), 100)
laplace.pdf(x,loc,scale)
```

Where laplace.ppf is similar to "qlaplace()" function and laplace.pmf is similar to "dlaplace()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

laplace.ppf([0.001, 0.5, 0.999],loc,scale)
```

Similar to "plaplace()" function in r.

```{code-cell} ipython3
## Generate random numbers

laplace.rvs(loc,scale,size=10)
```

Similar to "rlaplace()" function in r.

+++

### Logistic

```{code-cell} ipython3
from scipy.stats import logistic
```

```{code-cell} ipython3
## Calculate the first four moments

loc, scale = 16, 3
mean, var, skew, kurt = logistic.stats(loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(logistic.ppf(0.99), 100)
logistic.pdf(x,loc, scale)
```

Where logistic.ppf is similar to "qlogis()" function and logistic.pmf is similar to "dlogis()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

logistic.ppf([0.001, 0.5, 0.999],loc, scale)
```

Similar to "plogis()" function in r.

```{code-cell} ipython3
## Generate random numbers

logistic.rvs(loc, scale,size=10)
```

Similar to "rlogis()" function in r.

+++

### Noncentral Chi-square

```{code-cell} ipython3
from scipy.stats import ncx2
```

```{code-cell} ipython3
## Calculate the first four moments

df, nc = 30, 0.88
mean, var, skew, kurt = ncx2.stats(df, nc, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(ncx2.ppf(0.99, df, nc), 100)
ncx2.pdf(x, df, nc)
```

```{code-cell} ipython3
## Cummulative distribution function

ncx2.ppf([0.001, 0.5, 0.999], df, nc)
```

```{code-cell} ipython3
## Generate random numbers

ncx2.rvs(df, nc, size=10)
```

The corresponding r codes are same as the central chisquare.

+++

### Noncentral F

```{code-cell} ipython3
from scipy.stats import ncf
```

```{code-cell} ipython3
## Calculate the first four moments

dfn, dfd, nc = 18, 23, 0.342
mean, var, skew, kurt = ncf.stats(dfn, dfd, nc, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(ncf.ppf(0.99, dfn, dfd, nc), 100)
ncf.pdf(x, dfn, dfd, nc)
```

```{code-cell} ipython3
## Cummulative distribution function

ncf.ppf([0.001, 0.5, 0.999], dfn, dfd, nc)
```

```{code-cell} ipython3
## Generate random numbers

ncf.rvs(dfn, dfd, nc, size=10)
```

### Noncentral t

```{code-cell} ipython3
from scipy.stats import nct
```

```{code-cell} ipython3
## Calculate the first four moments

df, nc = 22, 0.13
mean, var, skew, kurt = nct.stats(df, nc, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(nct.ppf(0.99, df, nc), 100)
nct.pdf(x, df, nc)
```

```{code-cell} ipython3
## Cummulative distribution function

nct.ppf([0.001, 0.5, 0.999], df, nc)
```

```{code-cell} ipython3
## Generate random numbers

nct.rvs(df, nc, size=10)
```

### Normal

```{code-cell} ipython3
from scipy.stats import norm
```

```{code-cell} ipython3
## Calculate the first four moments

loc, scale = 10,2.4
mean, var, skew, kurt = norm.stats(loc,scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(norm.ppf(0.99), 100)
norm.pdf(x,loc,scale)
```

Where norm.ppf is similar to "qnorm()" function and norm.pmf is similar to "dnorm()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

norm.ppf([0.001, 0.5, 0.999],loc,scale)
```

Similar to "pnorm()" function in r.

```{code-cell} ipython3
## Generate random numbers

norm.rvs(size=10)
```

Similar to "rnorm()" function in r.

+++

### Pareto

```{code-cell} ipython3
from scipy.stats import pareto
```

```{code-cell} ipython3
## Calculate the first four moments

b = 5
mean, var, skew, kurt = pareto.stats(b, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(pareto.ppf(0.99, b), 100)
pareto.pdf(x, b)
```

Where pareto.ppf is similar to "qpareto()" function and pareto.pmf is similar to "dpareto()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

pareto.ppf([0.001, 0.5, 0.999], b)
```

Similar to "ppareto()" function in r.

```{code-cell} ipython3
## Generate random numbers

pareto.rvs(b, size=10)
```

Similar to "rpareto()" function in r.

+++

### Rayleigh

```{code-cell} ipython3
from scipy.stats import rayleigh
```

```{code-cell} ipython3
## Calculate the first four moments

loc, scale = 16, 3
mean, var, skew, kurt = rayleigh.stats(loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(rayleigh.ppf(0.99), 100)
rayleigh.pdf(x,loc, scale)
```

Where rayleigh.ppf is similar to "qrayleigh()" function and rayleigh.pmf is similar to "drayleigh()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

rayleigh.ppf([0.001, 0.5, 0.999],loc, scale)
```

Similar to "prayleigh()" function in r.

```{code-cell} ipython3
## Generate random numbers

rayleigh.rvs(loc, scale,size=10)
```

Similar to "rrayleigh()" function in r.

+++

### Student t

```{code-cell} ipython3
from scipy.stats import t
```

```{code-cell} ipython3
## Calculate the first four moments

df,loc,scale = 5, 12, 3 
mean, var, skew, kurt = t.stats(df,loc,scale, moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(t.ppf(0.99, df), 100)
t.pdf(x, df,loc,scale)
```

Where t.ppf is similar to "qt()" function and t.pmf is similar to "dt()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

t.ppf([0.001, 0.5, 0.999], df,loc,scale)
```

Similar to "pt()" function in r.

```{code-cell} ipython3
## Generate random numbers

t.rvs(df, size=10)
```

Similar to "rt()" function in r.

+++

### Uniform

```{code-cell} ipython3
from scipy.stats import uniform
```

```{code-cell} ipython3
## Calculate the first four moments

loc,scale=5,2
mean, var, skew, kurt = uniform.stats(loc,scale,moments='mvsk')
print(mean, var, skew, kurt)
```

```{code-cell} ipython3
## Probability density function

x = np.linspace(uniform.ppf(0.99), 100)
uniform.pdf(x,loc,scale)
```

Where uniform.ppf is similar to "qunif()" function and uniform.pmf is similar to "dunif()" function in r.

```{code-cell} ipython3
## Cummulative distribution function

uniform.ppf([0.001, 0.5, 0.999])
```

Similar to "punif()" function in r.

```{code-cell} ipython3
## Generate random numbers

uniform.rvs(size=10)
```

Similar to "runif()" function in r.

+++

Also there are several other continuous distributions under scipy.stats.
