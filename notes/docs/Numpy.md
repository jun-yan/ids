---
jupytext:
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

<!-- (about_py)= -->

# Numpy

By default Python does not have a concept of Arrays. And there is no inbuilt support for multidimensional arrays.

Python Numpy is a library that handles multidimensional arrays with ease. It has a great collection of functions that makes it easy while working with arrays.

It provides a multidimensional array object, and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data inside them. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren’t homogeneous.

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. 

The NumPy API is used extensively in **Pandas, SciPy, Matplotlib, scikit-learn, scikit-image** and most other data science and scientific Python packages.

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
## Save as csv

np.savetxt('new_data.csv', y1)
```

```{code-cell} ipython3
np.loadtxt('new_data.csv')
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

## Fancy Indexing

Fancy indexing means passing an array of indices to access multiple array elements at once

Remamber the simple indexing we'v seen before? Let's replace the singlel scaler with arrays of indices.

###  fancy indexing in one dimensional array:

```{code-cell} ipython3
import numpy as np
x = np.random.randint(100, size=10)
print(x)
```

    [64 90 19 13 66 37 12 38 64 82]

```{code-cell} ipython3
# instead of simple indexing like these:
[x[3], x[7], x[4], x[4]]
```

    [13, 38, 66, 66]

```{code-cell} ipython3
# we can now do this
x[[3, 7, 4, 4]]
```

    array([13, 38, 66, 66])

```{code-cell} ipython3
ind = np.array([[3, 7],
                     [4, 4]])
x[ind]
```

    array([[13, 38],
           [66, 66]])



With fancy indexing, the shape of the result reflects the shape <br />
of the index arrays rather than the shape of the array being indexed

### fancy indexing in multiple dimensional arrays:

Let's work in multiple dimensions

```{code-cell} ipython3
X = np.arange(12).reshape(3, 4)
X
```

    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

```{code-cell} ipython3
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

#  Fancy indexing reflects the broadcasted shape of the indeces.
```

    array([ 2,  5, 11])



What if we want to select all 3 rows and 3 columns specified?

```{code-cell} ipython3
row[:, np.newaxis] 

# one dimensional array becomes two dimensional array, 3 by 1, i.e., a column vector
```

    array([[0],
           [1],
           [2]])



Remember numpy.newaxis is used to increase the dimension of the existing array by one more dimension when used once?

1D array will become 2D array
2D array will become 3D array
3D array will become 4D array......

```{code-cell} ipython3
X[row[:, np.newaxis], col] 

## This is what we want
```

    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])

```{code-cell} ipython3
row[:, np.newaxis] * col
```

    array([[0, 0, 0],
           [2, 1, 3],
           [4, 2, 6]])



### Combine fancy indexing with slicing, simple indexing and mask

```{code-cell} ipython3
X[: , [2,1,3]] 

# Or we can combine fancy indexing with slicing.
```

    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])

```{code-cell} ipython3
X[1 , [2,1,3]] 

# we can combine fancy indexing with simple indexing.
```

    array([6, 5, 7])

```{code-cell} ipython3
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]

# There are only 1st column and 3rd column now
```

    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]])



### Modify values with fancy indexing

```{code-cell} ipython3
x = np.zeros(10)
print(x)

ind = np.array([0,1,2,3])
x[ind] = 9
print(x)

x[ind] -= 1 # minus 1 to the previous value
print(x)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [9. 9. 9. 9. 0. 0. 0. 0. 0. 0.]
    [8. 8. 8. 8. 0. 0. 0. 0. 0. 0.]


Repeated indices with these operations can cause some unexpected results.

```{code-cell} ipython3
x = np.zeros(10)
x[[0, 0, 0, 0]] = [2, 4, 6, 8]
print(x)

# multiple modifies at index 0, only the last one shows
```

    [8. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

```{code-cell} ipython3
x = np.zeros(10)
x[ [1,3,3,5,5,5] ] += 1
x
```

    array([0., 1., 0., 1., 0., 1., 0., 0., 0., 0.])



Why didn't we get x[3] = 2 and x[5] = 3?

Because under the hook, x[i] +1 is evaluated and the resaul is assigned to the index in x. <br />
So it is the assignment , not the augument, that happens mulitple times. <br />
To make the operation happen repeatedly, use '.at()':

```{code-cell} ipython3
x = np.zeros(10)
ind =  [1,3,3,5,5,5]
np.add.at(x, ind, 1)
print(x)
```

    [0. 1. 0. 2. 0. 3. 0. 0. 0. 0.]


## 2. Sorting

```{code-cell} ipython3
## First, let's get an array to be sorted

import random
x = random.sample(range(10), 10)
x
```

    [2, 8, 5, 0, 6, 7, 3, 4, 1, 9]

```{code-cell} ipython3
np.sort(x)

## Equivalent to sort() in R
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

```{code-cell} ipython3
np.argsort(x) # Returns the indices,

#  Equivalent to order() in R, though start with index 0.
```

    array([3, 8, 0, 6, 7, 2, 4, 5, 1, 9])

```{code-cell} ipython3
x

# x remains the same
```

    [2, 8, 5, 0, 6, 7, 3, 4, 1, 9]



We can see that np.sort() does not modify the input. <br />

To sort directly in place, we can use the sort method of arrays:

```{code-cell} ipython3
x.sort() # To sort directly in place
print(x)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


### Sorting along rows or columns

```{code-cell} ipython3
X = np.random.randint(1, 10, (3,5)) 
print(X)
```

    [[5 5 7 1 8]
     [4 3 9 6 6]
     [8 9 9 7 1]]

```{code-cell} ipython3
# sort each column of X

np.sort(X, axis=0)
```

    array([[4, 3, 7, 1, 1],
           [5, 5, 9, 6, 6],
           [8, 9, 9, 7, 8]])



Remember "axis" specifies the dimension of the array that will be collapsed, rather than the dimension that will be returned <br />

so axis=0 means the first axis(row) will be collapsed, which means the values within each columns will be sorted.

```{code-cell} ipython3
# sort each row of X

np.sort(X, axis=1)
```

    array([[1, 5, 5, 7, 8],
           [3, 4, 6, 6, 9],
           [1, 7, 8, 9, 9]])



### partitioning

```{code-cell} ipython3
x = np.array([9, 10, 8, 4, 0, 6, 2, 6, 5, 1, 0])
np.partition(x, 4)
```

    array([ 0,  1,  0,  2,  4,  5,  6,  6, 10,  8,  9])



The result is a new array with the smallest 3 values to the left of the partition, <br />
and the remaining values to the right.

Within each group, the order is arbituray.

```{code-cell} ipython3
## we can partition along any axis of a multidimensional array

X = np.random.randint(1, 10, (3,5)) 
np.partition(X, 2, axis=1)
```

    array([[3, 5, 7, 9, 9],
           [1, 3, 7, 9, 8],
           [1, 2, 8, 9, 8]])



## 3. Structured Data: NumPy’s Structured Arrays

extend from homogeneous data to compound, heterogeneous data.

### Creating a structured array

```{code-cell} ipython3
x = np.zeros(4, dtype=int)
x

# Just like creating a simple array
```

    array([0, 0, 0, 0])

```{code-cell} ipython3
# Creating a structured array : the dictionary method
data = np.zeros(4, dtype={'names':('name', 'cyl', 'mpg'),
                                     'formats':('U20', 'i4', 'f8')})
print(data)
print(data.dtype)
```

    [('', 0, 0.) ('', 0, 0.) ('', 0, 0.) ('', 0, 0.)]
    [('name', '<U20'), ('cyl', '<i4'), ('mpg', '<f8')]


'U10' translates to “Unicode string of maximum length 10” 

'i4' translates to “4-byte (i.e., 32 bit) signed integer” 

and 'f8' translates to “8-byte (i.e., 64 bit) float” ---See more by searching "Numpy data types"

```{code-cell} ipython3
data['name'] =  ['Honda Civic', 'Toyota Corolla', 'Ferrari Dino', 'Lincoln Continental']
data['cyl'] =  [4, 4, 6, 8]
data['mpg'] = [30.4, 33.9, 19.7, 10.4]
print(data)
```

    [('Honda Civic', 4, 30.4) ('Toyota Corolla', 4, 33.9)
     ('Ferrari Dino', 6, 19.7) ('Lincoln Continental', 8, 10.4)]


A compound type can also be specified as a list of tuples:

```{code-cell} ipython3
# Creating a structured array : the list-of-tuple method

data1 = np.zeros(4, dtype=[('name', 'U20'), ('cyl', 'i4'), ('mpg', 'f8')])
data1['name'] =  ['Honda Civic', 'Toyota Corolla', 'Ferrari Dino', 'Lincoln Continental']
data1['cyl'] =  [4, 4, 6, 8]
data1['mpg'] = [30.4, 33.9, 19.7, 10.4]
print(data1)

## This generate a same result
```

    [('Honda Civic', 4, 30.4) ('Toyota Corolla', 4, 33.9)
     ('Ferrari Dino', 6, 19.7) ('Lincoln Continental', 8, 10.4)]


### Refer to values in structured arrays

```{code-cell} ipython3
# By name: get all names

data['name']
```

    array(['Honda Civic', 'Toyota Corolla', 'Ferrari Dino',
           'Lincoln Continental'], dtype='<U20')

```{code-cell} ipython3
# By index: get first row of data 

data[0]
```

    ('Honda Civic', 4, 30.4)

```{code-cell} ipython3
# By index and name: get the name from the last row 

data[-1]['name']
```

    'Lincoln Continental'

```{code-cell} ipython3
# Using Boolean masking to filter the data on age 

data[data['cyl'] < 5]['name']
```

    array(['Honda Civic', 'Toyota Corolla'], dtype='<U20')



### record arrays

NumPy also provides the np.recarray class <br />
The only difference is that, fields can be accessed as attributes rather than dictionary keys

```{code-cell} ipython3
# Previously we access cyl by dictionary keys:
data['cyl']
```

    array([4, 4, 6, 8], dtype=int32)

```{code-cell} ipython3
# View the data as record array:

data = data.view(np.recarray)
data.cyl
```

    array([4, 4, 6, 8], dtype=int32)



### Let's move onto Pandas!

Pandas provides a DataFrame object, which is a structure built on NumPy arrays <br />
that offers a variety of useful data manipulation functionality <br />
similar to what we’ve shown here, as well as much, much more.
