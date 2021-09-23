<!-- #region -->
# Optimization

SciPy optimize provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints.  
It includes solvers for nonlinear problems (with support for both local and global optimization algorithms),  
linear programing, constrained and nonlinear least-squares, root finding, and curve fitting.  


## Least-squares minimization (least_squares)

SciPy is capable of solving robustified bound-constrained nonlinear least-squares problems:  
$$min \frac{1}{2} \Sigma_{1}^{m}\rho(f_{i}(x)^2)$$  
Subject to $$lb\leq x \leq ub$$


### Example of solving a fitting problem

$f_i(x)=\frac{1}{(xu+1)}-y_i, i=1,...,10$  
where $y_i$ are measurement values.The unknown vector of parameters is $x$.  
It is recommended to compute Jacobian matrix in a closed form:  
$J_i=\frac{-u}{(xu+1)^2}$  

<!-- #endregion -->

```python
from scipy.optimize import least_squares
import numpy as np
```


```python
def model(x, u):
    return 1/(x*u+1)
```


```python
def fun(x, u, y):
    return model(x,u) - y
```


```python
def jac(x,u,y):
    J = np.empty((u.size, x.size))
    J[:, 0] = -u/(x[0]*u+1)**2
    return J
```


```python
u = np.array([4.0, 2.0, 1.0, 2.5e-1])
y = np.array([0.2, 0.33, 0.5,0.8])
x0 = np.array([0.1])
res = least_squares(fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=1)
```

```python
res.x
```
```python
%config InlineBackend.figure_formats = ['svg']
import matplotlib.pyplot as plt # import plot package
u_test = np.linspace(0, 5)
y_test = model(res.x, u_test)
plt.plot(u, y, 'o', markersize=4, label='data')
plt.plot(u_test, y_test, label='fitted model')
plt.xlabel("u")
plt.ylabel("y")
plt.legend(loc='lower left')
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


```python
from scipy.optimize import minimize_scalar
f = lambda x: (x - 2) * (x + 1)**2
res = minimize_scalar(f, method='brent')
print(res.x)
```

```python
x_test = np.linspace(0, 5)
f_test = f(u_test)
plt.plot(x_test, f_test, label='f(x)')
plt.plot(1,f(1),'bo',label="min")
plt.xlabel("x")
plt.ylabel("f")
plt.legend(loc='lower right')
plt.show()
```

![svg](Optimization_files/Optimization_11_0.svg)
    

