## Numpy

By default Python does not have a concept of Arrays. And there is no inbuilt support for multidimensional arrays.

Python Numpy is a library that handles multidimensional arrays with ease. It has a great collection of functions that makes it easy while working with arrays.

It provides a multidimensional array object, and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data inside them. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren’t homogeneous.

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. 

The NumPy API is used extensively in **Pandas, SciPy, Matplotlib, scikit-learn, scikit-image** and most other data science and scientific Python packages.

### Array

An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. 

The rank of the array is the number of dimensions. The shape of the array is a tuple of integers giving the size of the array along each dimension.


```python
## Import the Numpy package

import numpy as np
```


```python
## Create Array

x = np.array([2,4,6,8])
x
```


```python
## Creating Zeroes

np.zeros(3)
```


```python
## Creating ones

np.ones(5)
```


```python
## Creating first 5 integers

np.arange(5)
```


```python
## Creating integers based in an interval with spacing

np.arange(1, 10, 3)
```


```python
## Creating an Array in a linearspace

np.linspace(0, 5, num=3)
```


```python
## Creating an identity matrix
np.eye(5)
```


```python
## Sort the array

ar = np.array([7,9,2,6])
np.sort(ar)
```


```python
## Merge arrays

x=np.array([2,4,6])
y=np.array([8,10,11])
np.concatenate((x,y))
```


```python
## Reshaping an array

a = np.arange(4)
print(a)
```


```python
b = a.reshape(2, 2)
print(b)
```


```python
## Indexing and Slicing

data = np.array([1, 5, 9, 2])
data[1]
```


```python
data[0:2]
```


```python
data[1:]
```


```python
data[::2]
```


```python
## Reverse elements in an array

data[::-1] 
```


```python
data[2::-1]
```


```python
a = np.array([[2, 4], [6, 8], [10, 12]])
print(a[a < 8])
```


```python
greater = (a >= 6)
print(a[greater])
```


```python
divisible_by_4 = a[a%4==0]
print(divisible_by_4)
```


```python
range = a[(a > 2) & (a < 12)]
print(range)
```


```python
up = (a > 6) | (a == 6)
print(a[up])
```


```python
#Stack Arrays
a1 = np.array([[1, 2],
               [3, 4]])

a2 = np.array([[5, 6],
               [7, 8]])

np.vstack((a1, a2))
```

Similar to function "rbind" in r.


```python
np.hstack((a1, a2))
```

Similar to function "cbind" in r.


```python
## Split Arrays

x = np.arange(4).reshape((2, 2))
x
```


```python
x1, x2 = np.vsplit(x, [1])
print(x1)
print(x2)
```


```python
x1, x2 = np.hsplit(x, [1])
print(x1)
print(x2)
```


```python
## Array functions

data = np.array([2, 4])
ones = np.ones(2, dtype=int)
data + ones
```


```python
data - ones
```


```python
data * data
```


```python
data / 3
```


```python
data//3
```


```python
print("-data = ", -data)
```


```python
## Power

print("data ** 2 = ", data ** 2)
```


```python
## Modulus

print("data % 4 = ", data % 4)
```


```python
## Summing an Array

a = np.array([2, 2, 2, 4])
a.sum()
```


```python
## Summing over rows

b = np.array([[0, 1], [2, 3]])
b.sum(axis=0)
```


```python
## Summing over Columns

b.sum(axis=1)
```


```python
## Minimum 

b.min()
```


```python
## Maximum

b.max()
```


```python
b.max(axis=0)
```


```python
b.max(axis=1)
```


```python
## Absolute Values

x = np.array([-2, -1, 0, 3, -4])
np.absolute(x)
```


```python
## Aggregates

x = np.arange(1, 5)
np.add.reduce(x)
```


```python
np.multiply.reduce(x)
```


```python
np.add.accumulate(x)
```


```python
np.multiply.accumulate(x)
```


```python
np.multiply.outer(x, x)
```


```python
## Random number generation

rng = np.random
rng.random(3)
```


```python
## Pulling unique values

a = np.array([1, 1, 2, 3, 4, 5, 2, 3, 1, 4, 8, 9])
np.unique(a)
```


```python
np.unique(a, return_index=True)
```


```python
np.unique(a, return_counts=True)
```


```python
## Transpose of a matrix
b
```


```python
np.transpose(b)
```


```python
## Flip array

x=np.arange(6)
x
```


```python
np.flip(x)
```


```python
y=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
np.flip(y)
```


```python
np.flip(y, axis=0)
```


```python
np.flip(y, axis=1)
```


```python
y[1]=np.flip(y[1])
print(y)
```


```python
y[:,1] = np.flip(y[:,1])
print(y)
```


```python
## Flattening an multidimensional array

y=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y.flatten()
```


```python
y1 = y.flatten()
y1[0] = 99
print(y) 
```


```python
print(y1)
```


```python
y1 = y.ravel()
y1[0] = 99
print(y) 
```


```python
print(y1)
```


```python
## Save and Load

np.save('data', y1)
```


```python
np.load('data.npy')
```


```python
## Save as csv

np.savetxt('new_data.csv', y1)
```


```python
np.loadtxt('new_data.csv')
```


```python
## Copy array

y2=y1.copy()
y2
```


```python
## Dot product

a = 2
b = 6
np.dot(a,b)
```


```python
A = np.array([1, 2, 3, 4])
B = np.array([5, 6, 7, 8])
np.dot(A, B)
```


```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)
```


```python
## Cross product

A = np.array([1, 2])
B = np.array([3, 4])
np.cross(A, B)
```


```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
np.cross(A, B)
```


```python
## Square root

A = [4, 9, 16, 1, 25]
np.sqrt(A)
```


```python
x = [4+1j, 9+16j]
np.sqrt(x)
```


```python
y = [-4, 9]
np.sqrt(y)
```


```python
## Average

a = np.array([1, 2, 3, 4]).reshape(2,2)
np.average(a)
```


```python
np.average(a, axis=0)
```


```python
np.average(a, axis=1)
```

Can perform this using "np.mean" function too.


```python
## Mean

np.mean(a)
```


```python
## Standard Deviation

np.std(a)
```


```python
np.std(a,axis=1)
```


```python
np.percentile(a, 25)
```


```python
np.median(a)
```


```python
np.percentile(a, 75)
```


```python
## Converting from array to list

a.tolist()
```


```python
## Converting from list to array

y=list([1, 2, 3, 4, 5])
np.array(y)
```


```python
ar=np.array([[True,True],[False,False]])
np.any(ar)
```


```python
## Check elements in an array is true

ar=np.array([[True,True],[True,True]])
np.all(ar)
```


```python
ar = np.array([[True,True],[False,False]])
np.all(ar)
```


```python
ar = np.array([[True,True], [True,False], [True,False]])
np.all(ar, axis=1)
```


```python
## Trignometric functions 
    
theta = np.linspace(1, np.pi, 2)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))
```


```python
## Inverse trignometric functions

x=[-1,0]
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))
```


```python
## Exponentials

x = [1, 3]
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("4^x =", np.power(4, x))
```


```python
## Logarithms

x = [1, 2, 3]
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))
```


```python
## More precision for small inputs

x = [0.001, 0.01]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))
```
