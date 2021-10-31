---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Optimization

SciPy optimize provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints.  
It includes solvers for nonlinear problems (with support for both local and global optimization algorithms),  
linear programing, constrained and nonlinear least-squares, root finding, and curve fitting.  


## Least-squares minimization (least_squares)

SciPy is capable of solving robustified bound-constrained nonlinear least-squares problems:  

$$
\text{min}\  \frac{1}{2} \sum\limits_{1}^{m}\rho(f_{i}(x_i;\theta)^2)
$$
  
Subject to

$$
lb\leq \theta \leq ub
$$

Where $x_i$ is predictor, $f(x_i)$ is the residual,   
$\rho(f_{i}(x_i;\theta)^2)$ is to reduce the influence of outliers on the solution


### Example of solving a fitting problem with 1000 samples

$f_i(x_i;\theta)=\frac{1}{(x_{i}\theta+1)}-y_i, i=1,...,n$  
where $y_i$ are measurement values.The unknown vector of parameters is $\theta$.  
It is recommended to compute derivative in a closed form:  
$J_i=\frac{-x_i}{(\theta*x_i+1)^2}$  

<!-- #endregion -->

```{code-cell} ipython3
from scipy.optimize import least_squares
import numpy as np
```

```{code-cell} ipython3
def model( theta,x):
    return 1/(x*theta+1)
```

```{code-cell} ipython3
def fun( theta,x, y):
    return y-model(x,theta)
```

```{code-cell} ipython3
def jac(theta,x,y):
    J = np.empty((x.size, theta.size))
    J[:, 0] = x/(theta[0]*x+1)**2
    return J
```

```{code-cell} ipython3
x = np.linspace(0,50,1000)
y = 1/(x+1)+np.random.normal(size=1000)
theta0 = np.array([0.1])
res = least_squares(fun, theta0, jac=jac, bounds=(0, 100),xtol=1e-10, args=(x, y), verbose=1)
```

```{code-cell} ipython3
res
```

```{code-cell} ipython3
res.x
```

```{code-cell} ipython3
%config InlineBackend.figure_formats = ['svg']
import matplotlib.pyplot as plt # import plot package
x_test = np.linspace(0, 50)
y_test = model(res.x, x_test)
plt.plot(x, y, 'o', markersize=4, label='data')
plt.plot(x_test, y_test, label='fitted model')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right')
plt.show()
```

## Univariate function minimizers (minimize_scalar)

Often only the minimum of an univariate function (i.e., a function that takes a scalar as input) is needed.  
In these circumstances, other optimization techniques have been developed that can work faster.   
These are accessible from the minimize_scalar function, which proposes several algorithms.  

### Unconstrained minimization (method='brent')

There are, actually, two methods that can be used to minimize an univariate function: brent and golden,  
but golden is included only for academic purposes and should rarely be used.    
The brent method uses Brent’s algorithm for locating a minimum.   
Optimally, a bracket (the bracket parameter) should be given which contains the minimum desired.   
A bracket is a triple $(a,b,c)$such that $f(a)>f(b)<f(c)$and $a<b<c$ .   
If this is not given, then alternatively two starting points can be chosen and a bracket will be found from these points using a simple marching algorithm.   
If these two starting points are not provided, 0 and 1 will be used

```{code-cell} ipython3
from scipy.optimize import minimize_scalar
f = lambda x: (x - 2) * (x + 1)**2
res = minimize_scalar(f, method='brent')
print(res.x)
```

```{code-cell} ipython3
x_test = np.linspace(0, 5)
f_test = f(x_test)
plt.plot(x_test, f_test, label='f(x)')
plt.plot(1,f(1),'bo',label="min")
plt.xlabel("x")
plt.ylabel("f")
plt.legend(loc='lower right')
plt.show()
```

## Unconstrained minimization of multivariate scalar functions ([minimize](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize))

+++

The minimize function provides a common interface to unconstrained and constrained   
minimization algorithms for multivariate scalar functions in scipy.optimize.   

```{code-cell} ipython3
import numpy as np
from scipy.optimize import minimize
```

```{code-cell} ipython3
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
```

```{code-cell} ipython3
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
```

```{code-cell} ipython3
print(res.x)
```
