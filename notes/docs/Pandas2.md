### Missing Data

The difference between data found in many tutorials and data in the real world is that
real-world data is rarely clean and homogeneous. In particular, many interesting
datasets will have some amount of data missing. 


```python
## Example 1 of missing data

import numpy as np
import pandas as pd

data = np.array([1, None, 6, 8])
data
```

If we use functions like sum() or min() on an array if none is present then it will return an error.


```python
try:
    data.sum()
except TypeError as e:
    print(type(e))
    print(e)
```

When a "None" value is present in an array, any arithmetic operation done on the array will always result in an error. 


```python
## Example 2 of missing data

data1 = np.array([1, np.nan, 2, 6])
3 + np.nan
```


```python
 5 * np.nan
```

Regardless of the operation, the result of arithmetic with NaN will be another NaN.


```python
## Detecting null values

data = pd.Series([1, np.nan, 'example', None])
data.isnull()
```


```python
data[data.notnull()]
```


```python
## Dropping null values

data.dropna()
```


```python
data2 = pd.DataFrame([[2, np.nan, 5],[4, 1, 7],[np.nan, 0, 3]])
data2
```


```python
data2.dropna()
```

We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 


```python
data2.dropna(axis='columns')
```


```python
data2[3] = np.nan
data2
```


```python
data2.dropna(axis='columns',how='all')
```


```python
data2.dropna(axis='rows',thresh=3)
```


```python
## Filling null values

data = pd.Series([3, np.nan, 4, None], index=list('abcd'))
data
```


```python
data.fillna(0)
```


```python
##Forward fill

data.fillna(method='ffill')
```


```python
## Back-fill

data.fillna(method='bfill')
```


```python
data2.fillna(method='ffill', axis=1)
```

### Hierarchical Indexing

It is useful to store higher-dimensional data indexed by more than one or two keys. It can be done using hierarchical indexing which incorporate multiple index levels within a single index. In this way, higher-dimensional data can be compactly represented within the familiar one-dimensional Series and two-dimensional DataFrame objects.


```python
## A Multiple Indexed Series

index = [('California', 2000), ('California', 2015),('New York', 2000), ('New York', 2015),('Illinois', 2000), ('Illinois', 2015)] 
populations = [24343547, 54343326,18976457, 23557766,12736448, 76453456]
pop = pd.Series(populations, index=index)
pop
```


```python
 pop[('California', 2015):('Illinois', 2000)]
```


```python
## Selecting just data for year 2015

pop[[i for i in pop.index if i[1] == 2015]]
```

As similar operation can be done in a much more efficient way. Our tuple-based indexing is essentially a rudimentary multi-index, and the Pandas MultiIndex type gives us the type of operations we wish to have. 


```python
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
pop
```

Here the first two columns of the Series representation show the multiple index values, while the third column shows the data. Notice that some entries are missing in the first column: in this multi-index representation, any blank entry indicates the same value as the line above it.


```python
pop[:, 2015]
```

This syntax is much more convenient than the previous method.


```python
## Unstack

pop1 = pop.unstack()
pop1
```

The unstack() method will quickly convert a multiple indexed Series into a conventionally indexed DataFrame.


```python
## Stack

pop1.stack()
```

Why hierarchical indexing?

The reason is simple: just as we were able to use multi-indexing to represent two-dimensional data within a one-dimensional Series, we can also use it to represent data of three or more dimensions in a Series or DataFrame. Each extra level in a multi-index represents an extra dimension of data; taking advantage of this property gives us much more flexibility in the types of data we can represent. Concretely, we might want to add another column of demographic data for each state at each year with a MultiIndex this is as easy as adding another column to the DataFrame. 


```python
pop1 = pd.DataFrame({'total': pop,'under18': [3123445, 1234567,2346257, 2461346,6434785, 9876544]})
pop1
```


```python
func = pop1['under18'] / pop1['total']
func.unstack()
```

All ufuncs and other functions used in pandas can be used in hierarchical indices too.


```python
## Constructing Multiple index series

data3 = pd.DataFrame(np.random.rand(4, 2),index=[['a', 'b', 'b', 'b'], [1, 1, 2, 3]],columns=['data1', 'data2'])
data3
```

Similarly, if you pass a dictionary with appropriate tuples as keys, Pandas will automatically recognize this and use a MultiIndex by default.


```python
data = {('California', 2000): 12345678,
 ('California', 2015): 23456789,
 ('Illinois', 2000): 34567890,
 ('Illinois', 2015): 45678901,
 ('New York', 2000): 56789012,
 ('New York', 2015): 67890123}
pd.Series(data)
```


```python
## MultiIndex level names

pop.index.names = ['State', 'Year']
pop
```


```python
## MultiIndex for Rows and columns

index = pd.MultiIndex.from_product([[2010, 2011], [1, 2]],names=['Year', 'Visit'])
columns = pd.MultiIndex.from_product([['A', 'B', 'C'], ['HR', 'Temp']],names=['Subject', 'Type'])
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 30
health = pd.DataFrame(data, index=index, columns=columns)
health
```


```python
health['B']
```

### Indexing and Slicing a MultiIndex


```python
## Multiple indexed Series

pop['California', 2000]
```


```python
 pop['California']
```


```python
pop[pop > 22000000]
```


```python
 pop[['California', 'Illinois']]
```


```python
## Multiple indexed dataframe

health['A', 'HR']
```


```python
health.loc[:, ('A', 'HR')]
```


```python
health.iloc[:2, :2]
```


```python
idx = pd.IndexSlice
health.loc[idx[:, 1], idx[:, 'HR']]
```

### Rearranging Multi indices


```python
## Sort

index = pd.MultiIndex.from_product([['x', 'z', 'y'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```


```python
try:
    data['x':'y']
except KeyError as e:
    print(type(e))
    print(e)
```

Any function performed on an unsorted multiindex data will result in an error.


```python
data = data.sort_index()
data
```


```python
data['x':'y']
```


```python
## Unstack

pop.unstack(level=0)
```


```python
pop.unstack(level=1)
```


```python
## Stack

pop.unstack().stack()
```


```python
## Reset Index

pop2 = pop.reset_index(name='population')
pop2
```

Often when you are working with data in the real world, the raw input data looks like this and it’s useful to build a MultiIndex from the column values.


```python
## Set Index

pop2.set_index(['State', 'Year'])
```

### Data Aggregation


```python
mean = health.groupby(level='Year').mean()
mean
```


```python
mean.groupby(axis=1, level='Type').mean()
```
