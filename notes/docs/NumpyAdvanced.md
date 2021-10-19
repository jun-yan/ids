# Numpy Advanced

## Fancy Indexing

Fancy indexing means passing an array of indices to access multiple array elements at once

Remamber the simple indexing we'v seen before? Let's replace the singlel scaler with arrays of indices.

###  fancy indexing in one dimensional array:


```python
import numpy as np
x = np.random.randint(100, size=10)
print(x)
```

    [64  8 21 64 45 96 85 30 27 62]



```python
# instead of simple indexing like these:
[x[3], x[7], x[4], x[4]]
```




    [64, 30, 45, 45]




```python
# we can now do this
x[[3, 7, 4, 4]]
```




    array([64, 30, 45, 45])




```python
ind = np.array([[3, 7],
                     [4, 4]])
x[ind]
```




    array([[64, 30],
           [45, 45]])



With fancy indexing, the shape of the result reflects the shape <br />
of the index arrays rather than the shape of the array being indexed

### fancy indexing in multiple dimensional arrays:

Let's work in multiple dimensions


```python
X = np.arange(12).reshape(3, 4)
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

#  Fancy indexing reflects the broadcasted shape of the indeces.
```




    array([ 2,  5, 11])



What if we want to select all 3 rows and 3 columns specified?


```python
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


```python
X[row[:, np.newaxis], col] 

## This is what we want
```




    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])




```python
row[:, np.newaxis] * col
```




    array([[0, 0, 0],
           [2, 1, 3],
           [4, 2, 6]])



### Combine fancy indexing with slicing, simple indexing and mask


```python
X[: , [2,1,3]] 

# Or we can combine fancy indexing with slicing.
```




    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])




```python
X[1 , [2,1,3]] 

# we can combine fancy indexing with simple indexing.
```




    array([6, 5, 7])




```python
X
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]

# There are only 1st column and 3rd column now
```




    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]])



### Modify values with fancy indexing


```python
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


```python
x = np.zeros(10)
x[[0, 0, 0, 0]] = [2, 4, 6, 8]
print(x)

# multiple modifies at index 0, only the last one shows
```

    [8. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



```python
x = np.zeros(10)
x[ [1,3,3,5,5,5] ] += 1
x
```




    array([0., 1., 0., 1., 0., 1., 0., 0., 0., 0.])



Why didn't we get x[3] = 2 and x[5] = 3?

Because under the hook, x[i] +1 is evaluated and the resaul is assigned to the index in x. <br />
So it is the assignment , not the augument, that happens mulitple times. <br />
To make the operation happen repeatedly, use '.at()':


```python
x = np.zeros(10)
ind =  [1,3,3,5,5,5]
np.add.at(x, ind, 1)
print(x)
```

    [0. 1. 0. 2. 0. 3. 0. 0. 0. 0.]


## Sorting


```python
## First, let's get an array to be sorted

import random
x = random.sample(range(10), 10)
x
```




    [2, 0, 7, 3, 5, 6, 8, 9, 1, 4]




```python
np.sort(x)

## Equivalent to sort() in R
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.argsort(x) # Returns the indices,

#  Equivalent to order() in R, though start with index 0.
```




    array([1, 8, 0, 3, 9, 4, 5, 2, 6, 7])




```python
x

# x remains the same
```




    [2, 0, 7, 3, 5, 6, 8, 9, 1, 4]



We can see that np.sort() does not modify the input. <br />

To sort directly in place, we can use the sort method of arrays:


```python
x.sort() # To sort directly in place
print(x)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


### Sorting along rows or columns


```python
X = np.random.randint(1, 10, (3,5)) 
print(X)
```

    [[1 4 4 2 3]
     [6 6 2 4 7]
     [1 1 8 4 3]]



```python
# sort each column of X

np.sort(X, axis=0)
```




    array([[1, 1, 2, 2, 3],
           [1, 4, 4, 4, 3],
           [6, 6, 8, 4, 7]])



Remember "axis" specifies the dimension of the array that will be collapsed, rather than the dimension that will be returned <br />

so axis=0 means the first axis(row) will be collapsed, which means the values within each columns will be sorted. 


```python
# sort each row of X

np.sort(X, axis=1)
```




    array([[1, 2, 3, 4, 4],
           [2, 4, 6, 6, 7],
           [1, 1, 3, 4, 8]])



### partitioning


```python
x = np.array([9, 10, 8, 4, 0, 6, 2, 6, 5, 1, 0])
np.partition(x, 4)
```




    array([ 0,  1,  0,  2,  4,  5,  6,  6, 10,  8,  9])



The result is a new array with the smallest 3 values to the left of the partition, <br />
and the remaining values to the right.

Within each group, the order is arbituray.


```python
## we can partition along any axis of a multidimensional array

X = np.random.randint(1, 10, (3,5)) 
np.partition(X, 2, axis=1)
```




    array([[3, 3, 5, 8, 7],
           [4, 4, 4, 5, 7],
           [3, 4, 4, 9, 9]])



## Structured Data: NumPy’s Structured Arrays

extend from homogeneous data to compound, heterogeneous data.

### Creating a structured array


```python
x = np.zeros(4, dtype=int)
x

# Just like creating a simple array
```




    array([0, 0, 0, 0])




```python
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


```python
data['name'] =  ['Honda Civic', 'Toyota Corolla', 'Ferrari Dino', 'Lincoln Continental']
data['cyl'] =  [4, 4, 6, 8]
data['mpg'] = [30.4, 33.9, 19.7, 10.4]
print(data)
```

    [('Honda Civic', 4, 30.4) ('Toyota Corolla', 4, 33.9)
     ('Ferrari Dino', 6, 19.7) ('Lincoln Continental', 8, 10.4)]


A compound type can also be specified as a list of tuples:


```python
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


```python
# By name: get all names

data['name']
```




    array(['Honda Civic', 'Toyota Corolla', 'Ferrari Dino',
           'Lincoln Continental'], dtype='<U20')




```python
# By index: get first row of data 

data[0]
```




    ('Honda Civic', 4, 30.4)




```python
# By index and name: get the name from the last row 

data[-1]['name']
```




    'Lincoln Continental'




```python
# Using Boolean masking to filter the data on age 

data[data['cyl'] < 5]['name']
```




    array(['Honda Civic', 'Toyota Corolla'], dtype='<U20')



### record arrays

NumPy also provides the np.recarray class <br />
The only difference is that, fields can be accessed as attributes rather than dictionary keys


```python
# Previously we access cyl by dictionary keys:
data['cyl']
```




    array([4, 4, 6, 8], dtype=int32)




```python
# View the data as record array:

data = data.view(np.recarray)
data.cyl
```




    array([4, 4, 6, 8], dtype=int32)



### Let's move onto Pandas!

Pandas provides a DataFrame object, which is a structure built on NumPy arrays <br />
that offers a variety of useful data manipulation functionality <br />
similar to what we’ve shown here, as well as much, much more.
