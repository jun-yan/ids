# Optimization

SciPy optimize provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints. It includes solvers for nonlinear problems (with support for both local and global optimization algorithms), linear programing, constrained and nonlinear least-squares, root finding, and curve fitting.

## Least-squares minimization (least_squares)

### Example of solving a fitting problem

$f_i(x)=\frac{1}{(xu+1)}-y_i, i=1,...,10$  
where $y_i$ are measurement values.The unknown vector of parameters is $x$.  
It is recommended to compute Jacobian matrix in a closed form:  
$J_i=\frac{-u}{(xu+1)^2}$


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
![svg](Optimization_files/Optimization_11_0.svg)
    

