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


# Support vector machine

## SVM in python with `sklearn`
Take the following simulaiton data as an example:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import svm
```


```python
# simulation data
from sklearn.datasets import make_blobs
%config InlineBackend.figure_format = 'svg'
# make_blobs & make_classification can be used to generate multiclass simulation data
X, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=123)
plt.scatter(X[:,0],X[:,1],c=y)
```


```python
# Fitting a SVM
model_svc = svm.SVC(kernel='linear') # svc = "Support vector classifier"
model_svc.fit(X,y)
# The key is the "support vector"
model_svc.support_vectors_
```


```python
# Create a mesh to plot
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model_svc.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y)
```

## Linear SVM
Suppose the given data is $(\vec X,y)=(\vec x_1,y_1),...,(\vec x_n,y_n)$ and $y_i$ are 1 or -1, indicating different classes.\
Any hyperplane($w$ is the normal vector to the plane) with the form:

$$
w^Tx-b=0
$$

You can use these boundaries to separate the data:

$$
 w^Tx-b&=1\\
 w^Tx-b&=-1
$$

Anything above first boundary is label 1, and anything below second boundary is label 2.
Geometrically we want to maximize the distance between the boundaries, which is $\frac{2}{||w||}$.\
We also want to prevent points from falling into the margin, so we add the constraints:
$w^Tx_i-b\ge1, y_i=1 \quad w^Tx_i-b\le-1, y_i=-1$\
The optimization for linear SVM is:

$$
\min_w ||w||^2  \\
\text{s.t  } y_i(w^Tx_i-b)\ge 1
$$

The loss function is(known as hard-margin):

$$
||w||^2+\frac{\lambda}{2}\sum_i{[1-y_i(w^Tx_i-b)]}
$$

Also we can extend the linear SVM to cases when data are not linearly separable. The loss function becomes(known as soft-margin):

$$
||w||^2+\frac{\lambda}{2}\sum_i{[1-y_i(w^Tx_i-b)]_+}
$$

The lagrangian dual of the above is:

$$
\max f(c_1,...,c_n)=\sum_i{c_i}-\frac{1}{2}\sum_i\sum_j {y_ic_i(x_i^Tx_j)y_jc_j}\\
\text{s.t  } \sum_i{c_iy_i}=0, 0\le c_i\le1/2n\lambda
$$

## Kernel SVM
Sometimes the data cannot be separate by a hyperplane, look at the following example:


```python
# toy example for a non-linear separable data
from sklearn.datasets import make_circles
X1, y1 = make_circles(100,factor=.3,noise=.1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50)
```

If we project the data into higher dimension such a separate might be possible. For example, $\phi(x_1,x_2)=(x_1,x_2,z=x_1^2+x_2^2)$.


```python
# scatter plot for transformed data (x1,x2,z)
z1 = np.array(X1 ** 2).sum(1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[:,0], X1[:,1], z1, c=y1)

```

Suppose we want to find a transform function $\phi(\cdot)$ and the classification vector in the transformed space is

$$\sum{c_iy_i\phi(x_i)}$$

And the linear kernel in the linear SVM is replaced by the $\phi$ kernel:$k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle$
The optimization problem becomes:

$$
\max f(c_1,...,c_n)=\sum_i{c_i}-\frac{1}{2}\sum_i\sum_j {y_ic_ik(x_i,x_j)y_jc_j}\\
\text{s.t} \sum_i{c_iy_i}=0, 0\le c_i\le1/2n\lambda
$$

Check the different SVM methods with different decision functions [here](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)!

### Kernel
Kernels are symmetric functions in the form $K(x,y)$\
Examples of kernels:\
Linear kernel: $K(x,y)=x^Ty$\
Gaussian kernel(RBF): $K(x,y)=e^{-\frac{||x-y||^2}{2\sigma^2}}=e^{-\gamma||x-y||^2}$\
Laplacian kernel: $K(x,y)=e^{-\alpha||x-y||}$\
In `sklearn`, we can apply kernelized SVM by changing linear kernel to RBF kernel easily.
There are two parameters in RBF kernel: $c$ and $\gamma$. They are related to the smooth of decision surface and influence of single training sample. A [gridsearch CV](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) can be used to decide these parameters.


```python
clf = svm.SVC(kernel = 'rbf')
clf.fit(X1,y1)
```


```python
# Create a mesh to plot
h = .02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X1[:,0],X1[:,1],c=y1)
```

## More about `sklearn.svm`
There are functions for fitting a SVM classification/regression model. 
### Classification
- SVC: traing "one vs one" model for muticlassification.
- NuSVC: similar to SVC
- LinearSVC: faster, but does not accept parameter kernel; "one vs rest" for muticlassification

### Regression
- SVR
- NuSVR
- LinearSVR

### Probability
SVM does not dirrectly provide probability estimates, but it can be calculated using cross-validation.


```python
# Iris data simulation
from sklearn import datasets
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# scaling data is usually good for SVM, and model-fitting remains same under scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# search C and gamma
# For an initial search, a logarithmic grid with basis 10 is often helpful. 
# Using a basis of 2, a finer tuning can be achieved but at a much higher cost.
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = {'C': C_range,
              'gamma': gamma_range}
cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=1)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid,cv=cv)
grid.fit(X, y)

print("C_range: %s \ngamma_range: %s"% (C_range, gamma_range))
print(grid.best_params_)

```


```python
# visuallization
# only use 2 features and binary Y
X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1
X_2d = scaler.fit_transform(X_2d)

# train classifiers on a smaller c/gamma sets
C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))
        

# draw the decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='small')

    # visualize parameter's effect on decision function
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())


```


```python
# draw the decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    W = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    W = W.reshape(xx.shape)
    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='small')

    # visualize parameter's effect on decision function
    plt.contourf(xx, yy, W, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
```

### Suggestions about tuning SVM kernels
- `Gamma` is the parameter of the Gaussian radial basis function. `C` is the parameter for the soft margin cost function, which controls the influence of each individual support vector.
- The `gamma` parameter defines how far the influence of a single training example reaches, with low values meaning 'far' and high values meaning 'close'. When `gamma` is very small, the model is too constrained and cannot capture the complexity or "shape" of the data. 
- The `C` parameter trades off correct classification against the decision function's margin. A lower`C` will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. In other words `C` behaves as a regularization parameter in the SVM.
- In practice, a logarithmic grid from $10^{-3}$ to $10^3$ is usually sufficient. If the best parameters lie on the boundaries of the grid, it can be extended in that direction in a subsequent search.

# Reference
- [https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/svm/plot_svm_parameters_selection.html]
- [https://scikit-learn.org/stable/modules/svm.html]
