## Pandas: Missing Data and Hierarchical Indexing

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




    array([1, None, 6, 8], dtype=object)



If we use functions like sum() or min() on an array if none is present then it will return an error.


```python
try:
    data.sum()
except TypeError as e:
    print(type(e))
    print(e)
```

    <class 'TypeError'>
    unsupported operand type(s) for +: 'int' and 'NoneType'
    

When a "None" value is present in an array, any arithmetic operation done on the array will always result in an error. 


```python
## Example 2 of missing data

data1 = np.array([1, np.nan, 2, 6])
3 + np.nan
```




    nan




```python
 5 * np.nan
```




    nan



Regardless of the operation, the result of arithmetic with NaN will be another NaN.


```python
## Detecting null values

data = pd.Series([1, np.nan, 'example', None])
data.isnull()
```




    0    False
    1     True
    2    False
    3     True
    dtype: bool




```python
data[data.notnull()]
```




    0          1
    2    example
    dtype: object




```python
## Dropping null values

data.dropna()
```




    0          1
    2    example
    dtype: object




```python
data2 = pd.DataFrame([[2, np.nan, 5],[4, 1, 7],[np.nan, 0, 3]])
data2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 


```python
data2.dropna(axis='columns')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2[3] = np.nan
data2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2.dropna(axis='columns',how='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2.dropna(axis='rows',thresh=3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Filling null values

data = pd.Series([3, np.nan, 4, None], index=list('abcd'))
data
```




    a    3.0
    b    NaN
    c    4.0
    d    NaN
    dtype: float64




```python
data.fillna(0)
```




    a    3.0
    b    0.0
    c    4.0
    d    0.0
    dtype: float64




```python
##Forward fill

data.fillna(method='ffill')
```




    a    3.0
    b    3.0
    c    4.0
    d    4.0
    dtype: float64




```python
## Back-fill

data.fillna(method='bfill')
```




    a    3.0
    b    4.0
    c    4.0
    d    NaN
    dtype: float64




```python
data2.fillna(method='ffill', axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



### Hierarchical Indexing

It is useful to store higher-dimensional data indexed by more than one or two keys. It can be done using hierarchical indexing which incorporate multiple index levels within a single index. In this way, higher-dimensional data can be compactly represented within the familiar one-dimensional Series and two-dimensional DataFrame objects.


```python
## A Multiple Indexed Series

index = [('California', 2000), ('California', 2015),('New York', 2000), ('New York', 2015),('Illinois', 2000), ('Illinois', 2015)] 
populations = [24343547, 54343326,18976457, 23557766,12736448, 76453456]
pop = pd.Series(populations, index=index)
pop
```




    (California, 2000)    24343547
    (California, 2015)    54343326
    (New York, 2000)      18976457
    (New York, 2015)      23557766
    (Illinois, 2000)      12736448
    (Illinois, 2015)      76453456
    dtype: int64




```python
 pop[('California', 2015):('Illinois', 2000)]
```




    (California, 2015)    54343326
    (New York, 2000)      18976457
    (New York, 2015)      23557766
    (Illinois, 2000)      12736448
    dtype: int64




```python
## Selecting just data for year 2015

pop[[i for i in pop.index if i[1] == 2015]]
```




    (California, 2015)    54343326
    (New York, 2015)      23557766
    (Illinois, 2015)      76453456
    dtype: int64



As similar operation can be done in a much more efficient way. Our tuple-based indexing is essentially a rudimentary multi-index, and the Pandas MultiIndex type gives us the type of operations we wish to have. 


```python
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
pop
```




    California  2000    24343547
                2015    54343326
    New York    2000    18976457
                2015    23557766
    Illinois    2000    12736448
                2015    76453456
    dtype: int64



Here the first two columns of the Series representation show the multiple index values, while the third column shows the data. Notice that some entries are missing in the first column: in this multi-index representation, any blank entry indicates the same value as the line above it.


```python
pop[:, 2015]
```




    California    54343326
    New York      23557766
    Illinois      76453456
    dtype: int64



This syntax is much more convenient than the previous method.


```python
## Unstack

pop1 = pop.unstack()
pop1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2000</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>24343547</td>
      <td>54343326</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12736448</td>
      <td>76453456</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>23557766</td>
    </tr>
  </tbody>
</table>
</div>



The unstack() method will quickly convert a multiple indexed Series into a conventionally indexed DataFrame.


```python
## Stack

pop1.stack()
```




    California  2000    24343547
                2015    54343326
    Illinois    2000    12736448
                2015    76453456
    New York    2000    18976457
                2015    23557766
    dtype: int64



Why hierarchical indexing?

The reason is simple: just as we were able to use multi-indexing to represent two-dimensional data within a one-dimensional Series, we can also use it to represent data of three or more dimensions in a Series or DataFrame. Each extra level in a multi-index represents an extra dimension of data; taking advantage of this property gives us much more flexibility in the types of data we can represent. Concretely, we might want to add another column of demographic data for each state at each year with a MultiIndex this is as easy as adding another column to the DataFrame. 


```python
pop1 = pd.DataFrame({'total': pop,'under18': [3123445, 1234567,2346257, 2461346,6434785, 9876544]})
pop1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>total</th>
      <th>under18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>24343547</td>
      <td>3123445</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>54343326</td>
      <td>1234567</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
      <td>2346257</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>23557766</td>
      <td>2461346</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Illinois</th>
      <th>2000</th>
      <td>12736448</td>
      <td>6434785</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>76453456</td>
      <td>9876544</td>
    </tr>
  </tbody>
</table>
</div>




```python
func = pop1['under18'] / pop1['total']
func.unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2000</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>0.128307</td>
      <td>0.022718</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>0.505226</td>
      <td>0.129184</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>0.123640</td>
      <td>0.104481</td>
    </tr>
  </tbody>
</table>
</div>



All ufuncs and other functions used in pandas can be used in hierarchical indices too.


```python
## Constructing Multiple index series

data3 = pd.DataFrame(np.random.rand(4, 2),index=[['a', 'b', 'b', 'b'], [1, 1, 2, 3]],columns=['data1', 'data2'])
data3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>1</th>
      <td>0.922987</td>
      <td>0.498359</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">b</th>
      <th>1</th>
      <td>0.466856</td>
      <td>0.057370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.466726</td>
      <td>0.897659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.617695</td>
      <td>0.034962</td>
    </tr>
  </tbody>
</table>
</div>



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




    California  2000    12345678
                2015    23456789
    Illinois    2000    34567890
                2015    45678901
    New York    2000    56789012
                2015    67890123
    dtype: int64




```python
## MultiIndex level names

pop.index.names = ['State', 'Year']
pop
```




    State       Year
    California  2000    24343547
                2015    54343326
    New York    2000    18976457
                2015    23557766
    Illinois    2000    12736448
                2015    76453456
    dtype: int64




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Subject</th>
      <th colspan="2" halign="left">A</th>
      <th colspan="2" halign="left">B</th>
      <th colspan="2" halign="left">C</th>
    </tr>
    <tr>
      <th></th>
      <th>Type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2010</th>
      <th>1</th>
      <td>35.0</td>
      <td>30.9</td>
      <td>19.0</td>
      <td>30.2</td>
      <td>32.0</td>
      <td>31.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>29.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.0</td>
      <td>28.6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2011</th>
      <th>1</th>
      <td>30.0</td>
      <td>29.6</td>
      <td>36.0</td>
      <td>30.4</td>
      <td>18.0</td>
      <td>28.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47.0</td>
      <td>29.6</td>
      <td>25.0</td>
      <td>30.2</td>
      <td>10.0</td>
      <td>27.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
health['B']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2010</th>
      <th>1</th>
      <td>19.0</td>
      <td>30.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2011</th>
      <th>1</th>
      <td>36.0</td>
      <td>30.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.0</td>
      <td>30.2</td>
    </tr>
  </tbody>
</table>
</div>



### Indexing and Slicing a MultiIndex


```python
## Multiple indexed Series

pop['California', 2000]
```




    24343547




```python
 pop['California']
```




    Year
    2000    24343547
    2015    54343326
    dtype: int64




```python
pop[pop > 22000000]
```




    State       Year
    California  2000    24343547
                2015    54343326
    New York    2015    23557766
    Illinois    2015    76453456
    dtype: int64




```python
 pop[['California', 'Illinois']]
```




    State       Year
    California  2000    24343547
                2015    54343326
    Illinois    2000    12736448
                2015    76453456
    dtype: int64




```python
## Multiple indexed dataframe

health['A', 'HR']
```




    Year  Visit
    2010  1        35.0
          2        18.0
    2011  1        30.0
          2        47.0
    Name: (A, HR), dtype: float64




```python
health.loc[:, ('A', 'HR')]
```




    Year  Visit
    2010  1        35.0
          2        18.0
    2011  1        30.0
          2        47.0
    Name: (A, HR), dtype: float64




```python
health.iloc[:2, :2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Subject</th>
      <th colspan="2" halign="left">A</th>
    </tr>
    <tr>
      <th></th>
      <th>Type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2010</th>
      <th>1</th>
      <td>35.0</td>
      <td>30.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
idx = pd.IndexSlice
health.loc[idx[:, 1], idx[:, 'HR']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Subject</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th></th>
      <th>Type</th>
      <th>HR</th>
      <th>HR</th>
      <th>HR</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Visit</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <th>1</th>
      <td>35.0</td>
      <td>19.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2011</th>
      <th>1</th>
      <td>30.0</td>
      <td>36.0</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
</div>



### Rearranging Multi indices


```python
## Sort

index = pd.MultiIndex.from_product([['x', 'z', 'y'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```




    char  int
    x     1      0.916248
          2      0.386705
    z     1      0.810961
          2      0.207214
    y     1      0.877897
          2      0.646366
    dtype: float64




```python
try:
    data['x':'y']
except KeyError as e:
    print(type(e))
    print(e)
```

    <class 'pandas.errors.UnsortedIndexError'>
    'Key length (1) was greater than MultiIndex lexsort depth (0)'
    

Any function performed on an unsorted multiindex data will result in an error.


```python
data = data.sort_index()
data
```




    char  int
    x     1      0.916248
          2      0.386705
    y     1      0.877897
          2      0.646366
    z     1      0.810961
          2      0.207214
    dtype: float64




```python
data['x':'y']
```




    char  int
    x     1      0.916248
          2      0.386705
    y     1      0.877897
          2      0.646366
    dtype: float64




```python
## Unstack

pop.unstack(level=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>State</th>
      <th>California</th>
      <th>Illinois</th>
      <th>New York</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>24343547</td>
      <td>12736448</td>
      <td>18976457</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>54343326</td>
      <td>76453456</td>
      <td>23557766</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop.unstack(level=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Year</th>
      <th>2000</th>
      <th>2015</th>
    </tr>
    <tr>
      <th>State</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>24343547</td>
      <td>54343326</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12736448</td>
      <td>76453456</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>23557766</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Stack

pop.unstack().stack()
```




    State       Year
    California  2000    24343547
                2015    54343326
    Illinois    2000    12736448
                2015    76453456
    New York    2000    18976457
                2015    23557766
    dtype: int64




```python
## Reset Index

pop2 = pop.reset_index(name='population')
pop2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>California</td>
      <td>2000</td>
      <td>24343547</td>
    </tr>
    <tr>
      <th>1</th>
      <td>California</td>
      <td>2015</td>
      <td>54343326</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York</td>
      <td>2000</td>
      <td>18976457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>2015</td>
      <td>23557766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Illinois</td>
      <td>2000</td>
      <td>12736448</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Illinois</td>
      <td>2015</td>
      <td>76453456</td>
    </tr>
  </tbody>
</table>
</div>



Often when you are working with data in the real world, the raw input data looks like this and it’s useful to build a MultiIndex from the column values.


```python
## Set Index

pop2.set_index(['State', 'Year'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>population</th>
    </tr>
    <tr>
      <th>State</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>24343547</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>54343326</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>23557766</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Illinois</th>
      <th>2000</th>
      <td>12736448</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>76453456</td>
    </tr>
  </tbody>
</table>
</div>



### Data Aggregation


```python
mean = health.groupby(level='Year').mean()
mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Subject</th>
      <th colspan="2" halign="left">A</th>
      <th colspan="2" halign="left">B</th>
      <th colspan="2" halign="left">C</th>
    </tr>
    <tr>
      <th>Type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>26.5</td>
      <td>29.95</td>
      <td>24.5</td>
      <td>30.1</td>
      <td>28.0</td>
      <td>30.20</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>38.5</td>
      <td>29.60</td>
      <td>30.5</td>
      <td>30.3</td>
      <td>14.0</td>
      <td>28.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
mean.groupby(axis=1, level='Type').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>26.333333</td>
      <td>30.083333</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>27.666667</td>
      <td>29.416667</td>
    </tr>
  </tbody>
</table>
</div>


