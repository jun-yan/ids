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

# Numpy

By default Python does not have a concept of Arrays. And there is no inbuilt support for multidimensional arrays.

Python Numpy is a library that handles multidimensional arrays with ease. It has a great collection of functions that makes it easy while working with arrays.

It provides a multidimensional array object, and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data inside them. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren’t homogeneous.

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. 

The NumPy API is used extensively in **Pandas, SciPy, Matplotlib, scikit-learn, scikit-image** and most other data science and scientific Python packages.

+++

## Array

An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. 

The rank of the array is the number of dimensions. The shape of the array is a tuple of integers giving the size of the array along each dimension.

```{code-cell} ipython3
## Import the Numpy package

import numpy as np
```

```{code-cell} ipython3
## Create Array

x = np.array([2,4,6,8])
x
```

```{code-cell} ipython3
## Creating Zeroes

np.zeros(3)
```

```{code-cell} ipython3
## Creating ones

np.ones(5)
```

```{code-cell} ipython3
## Creating first 5 integers

np.arange(5)
```

```{code-cell} ipython3
## Creating integers based in an interval with spacing

np.arange(1, 10, 3)
```

```{code-cell} ipython3
## Creating an Array in a linearspace

np.linspace(0, 5, num=3)
```

```{code-cell} ipython3
## Creating an identity matrix
np.eye(5)
```

```{code-cell} ipython3
## Sort the array

ar = np.array([7,9,2,6])
np.sort(ar)
```

```{code-cell} ipython3
## Merge arrays

x=np.array([2,4,6])
y=np.array([8,10,11])
np.concatenate((x,y))
```

```{code-cell} ipython3
## Reshaping an array

a = np.arange(4)
print(a)
```

```{code-cell} ipython3
b = a.reshape(2, 2)
print(b)
```

```{code-cell} ipython3
## Indexing and Slicing

data = np.array([1, 5, 9, 2])
data[1]
```

```{code-cell} ipython3
data[0:2]
```

```{code-cell} ipython3
data[1:]
```

```{code-cell} ipython3
data[::2]
```

```{code-cell} ipython3
## Reverse elements in an array

data[::-1] 
```

```{code-cell} ipython3
data[2::-1]
```

```{code-cell} ipython3
a = np.array([[2, 4], [6, 8], [10, 12]])
print(a[a < 8])
```

```{code-cell} ipython3
greater = (a >= 6)
print(a[greater])
```

```{code-cell} ipython3
divisible_by_4 = a[a%4==0]
print(divisible_by_4)
```

```{code-cell} ipython3
range = a[(a > 2) & (a < 12)]
print(range)
```

```{code-cell} ipython3
up = (a > 6) | (a == 6)
print(a[up])
```

```{code-cell} ipython3
#Stack Arrays
a1 = np.array([[1, 2],
               [3, 4]])

a2 = np.array([[5, 6],
               [7, 8]])

np.vstack((a1, a2))
```

Similar to function "rbind" in r.

```{code-cell} ipython3
np.hstack((a1, a2))
```

Similar to function "cbind" in r.

```{code-cell} ipython3
## Split Arrays

x = np.arange(4).reshape((2, 2))
x
```

```{code-cell} ipython3
x1, x2 = np.vsplit(x, [1])
print(x1)
print(x2)
```

```{code-cell} ipython3
x1, x2 = np.hsplit(x, [1])
print(x1)
print(x2)
```

```{code-cell} ipython3
## Array functions

data = np.array([2, 4])
ones = np.ones(2, dtype=int)
data + ones
```

```{code-cell} ipython3
data - ones
```

```{code-cell} ipython3
data * data
```

```{code-cell} ipython3
data / 3
```

```{code-cell} ipython3
data//3
```

```{code-cell} ipython3
print("-data = ", -data)
```

```{code-cell} ipython3
## Power

print("data ** 2 = ", data ** 2)
```

```{code-cell} ipython3
## Modulus

print("data % 4 = ", data % 4)
```

```{code-cell} ipython3
## Summing an Array

a = np.array([2, 2, 2, 4])
a.sum()
```

```{code-cell} ipython3
## Summing over rows

b = np.array([[0, 1], [2, 3]])
b.sum(axis=0)
```

```{code-cell} ipython3
## Summing over Columns

b.sum(axis=1)
```

```{code-cell} ipython3
## Minimum 

b.min()
```

```{code-cell} ipython3
## Maximum

b.max()
```

```{code-cell} ipython3
b.max(axis=0)
```

```{code-cell} ipython3
b.max(axis=1)
```

```{code-cell} ipython3
## Absolute Values

x = np.array([-2, -1, 0, 3, -4])
np.absolute(x)
```

```{code-cell} ipython3
## Aggregates

x = np.arange(1, 5)
np.add.reduce(x)
```

```{code-cell} ipython3
np.multiply.reduce(x)
```

```{code-cell} ipython3
np.add.accumulate(x)
```

```{code-cell} ipython3
np.multiply.accumulate(x)
```

```{code-cell} ipython3
np.multiply.outer(x, x)
```

```{code-cell} ipython3
## Random number generation

rng = np.random
rng.random(3)
```

```{code-cell} ipython3
## Pulling unique values

a = np.array([1, 1, 2, 3, 4, 5, 2, 3, 1, 4, 8, 9])
np.unique(a)
```

```{code-cell} ipython3
np.unique(a, return_index=True)
```

```{code-cell} ipython3
np.unique(a, return_counts=True)
```

```{code-cell} ipython3
## Transpose of a matrix
b
```

```{code-cell} ipython3
np.transpose(b)
```

```{code-cell} ipython3
## Flip array

x=np.arange(6)
x
```

```{code-cell} ipython3
np.flip(x)
```

```{code-cell} ipython3
y=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
np.flip(y)
```

```{code-cell} ipython3
np.flip(y, axis=0)
```

```{code-cell} ipython3
np.flip(y, axis=1)
```

```{code-cell} ipython3
y[1]=np.flip(y[1])
print(y)
```

```{code-cell} ipython3
y[:,1] = np.flip(y[:,1])
print(y)
```

```{code-cell} ipython3
## Flattening an multidimensional array

y=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y.flatten()
```

```{code-cell} ipython3
y1 = y.flatten()
y1[0] = 99
print(y) 
```

```{code-cell} ipython3
print(y1)
```

```{code-cell} ipython3
y1 = y.ravel()
y1[0] = 99
print(y) 
```

```{code-cell} ipython3
print(y1)
```

```{code-cell} ipython3
## Save and Load

np.save('data', y1)
```

```{code-cell} ipython3
np.load('data.npy')
```

```{code-cell} ipython3
#Deleting the created file

import os

os.remove('data.npy')
```

```{code-cell} ipython3
## Save as csv

np.savetxt('new_data.csv', y1)
```

```{code-cell} ipython3
np.loadtxt('new_data.csv')
```

```{code-cell} ipython3
#Deleting the created file

os.remove('new_data.csv')
```

```{code-cell} ipython3
## Copy array

y2=y1.copy()
y2
```

```{code-cell} ipython3
## Dot product

a = 2
b = 6
np.dot(a,b)
```

```{code-cell} ipython3
A = np.array([1, 2, 3, 4])
B = np.array([5, 6, 7, 8])
np.dot(A, B)
```

```{code-cell} ipython3
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)
```

```{code-cell} ipython3
## Cross product

A = np.array([1, 2])
B = np.array([3, 4])
np.cross(A, B)
```

```{code-cell} ipython3
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
np.cross(A, B)
```

```{code-cell} ipython3
## Square root

A = [4, 9, 16, 1, 25]
np.sqrt(A)
```

```{code-cell} ipython3
x = [4+1j, 9+16j]
np.sqrt(x)
```

```{code-cell} ipython3
y = [-4, 9]
np.sqrt(y)
```

```{code-cell} ipython3
## Average

a = np.array([1, 2, 3, 4]).reshape(2,2)
np.average(a)
```

```{code-cell} ipython3
np.average(a, axis=0)
```

```{code-cell} ipython3
np.average(a, axis=1)
```

Can perform this using "np.mean" function too.

```{code-cell} ipython3
## Mean

np.mean(a)
```

```{code-cell} ipython3
## Standard Deviation

np.std(a)
```

```{code-cell} ipython3
np.std(a,axis=1)
```

```{code-cell} ipython3
np.percentile(a, 25)
```

```{code-cell} ipython3
np.median(a)
```

```{code-cell} ipython3
np.percentile(a, 75)
```

```{code-cell} ipython3
## Converting from array to list

a.tolist()
```

```{code-cell} ipython3
## Converting from list to array

y=list([1, 2, 3, 4, 5])
np.array(y)
```

```{code-cell} ipython3
ar=np.array([[True,True],[False,False]])
np.any(ar)
```

```{code-cell} ipython3
## Check elements in an array is true

ar=np.array([[True,True],[True,True]])
np.all(ar)
```

```{code-cell} ipython3
ar = np.array([[True,True],[False,False]])
np.all(ar)
```

```{code-cell} ipython3
ar = np.array([[True,True], [True,False], [True,False]])
np.all(ar, axis=1)
```

```{code-cell} ipython3
## Trignometric functions 
    
theta = np.linspace(1, np.pi, 2)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))
```

```{code-cell} ipython3
## Inverse trignometric functions

x=[-1,0]
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))
```

```{code-cell} ipython3
## Exponentials

x = [1, 3]
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("4^x =", np.power(4, x))
```

```{code-cell} ipython3
## Logarithms

x = [1, 2, 3]
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))
```

```{code-cell} ipython3
## More precision for small inputs

x = [0.001, 0.01]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))
```
