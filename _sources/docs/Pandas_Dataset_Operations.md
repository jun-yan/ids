---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.12.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Pandas: Dataset Operations

Dataset can be combined in a multidue of different ways from other datasets. Operations that can be used can be from a straightforward approch such as using concatenation from two datasets. Alternatively, one could use database-style joins and merges to correctly handle any overlaps between datasets. 

`Series` and `DataFrames` both use the concatenation function, and Pandas includes functions and methods that make this sort of data wrangling fast and straightforward.

+++

## Simple Concatenation Using pd.concat

```{code-cell} ipython3
# !pip3 install pandas

import numpy as np
import pandas as pd

# pd. __version__

def create_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

create_df(['red', 'green', 'blue'], range(3))
```

`pd.concat()` can be used for simple concatenation of Series and DataFrames, it defaults to row-wise but can be specified to take place along any axis. 
  
Here is an example of row-wise concatination.

```{code-cell} ipython3
series1 = pd.Series(["red", "green", "blue"], [1, 2, 3])
series2 = pd.Series(["yellow", "orange", "purple"], [4, 5, 6])
series_concat = pd.concat([series1, series2])
print(series1); print(); 
print(series2); print(); 
print(series_concat); print() # final result
```

We can concatinate two DataFrames that have shared columns and merge the columns together, as seen here.

```{code-cell} ipython3
df1 = create_df(["red", "green"], [0, 1])
df2 = create_df(["red", "green"], [2, 3])
df_concat1 = pd.concat([df1, df2])
print(df1); print();
print(df2); print();
print(df_concat1); print()
```

We can also concatinate column-wise by passing the argument `axis = 1` or `axis = 'col'`.

```{code-cell} ipython3
df3 = create_df(["red", "green"], [0, 1])
df4 = create_df(["blue", "yellow"], [0, 1])
df_concat2 = pd.concat([df3, df4], axis = 1) # axis = 'col'
print(df3); print();
print(df4); print();
print(df_concat2)
```

### Concatinating with Duplicate Indices
`pd.concat()` can be used to handle situations where you concatinate with dulpicate indicies. By default it raises an error if this is done.

```{code-cell} ipython3
x = create_df(["red", "green"], [0, 1])
y = create_df(["red", "green"], [2, 3])
y.index = x.index

try:
    pd.concat([x, y], verify_integrity = True) # duplicated indices
except ValueError as e:
    print("ValueError:", e); print()
```

You can bypass this error by passing the argument `ignore_index = True` which will cause `concat` to ignore the indicies of the DataFrames it is concatinating.

```{code-cell} ipython3
df_concat3 = pd.concat([x, y], ignore_index = True)
print(x); print();
print(y); print();
print(df_concat3); print()
```

Another option is to pass an argument specifying keys. The `keys` argument must be a list, tuple, or some other sequence which corisponds to the DataFrames you are concatenating. Here we use `keys = ["x", "y"]` to add x and y as keys for the two DataFrames.

```{code-cell} ipython3
df_concat4 = pd.concat([x, y], keys = ["x", "y"])
print(x); print();
print(y); print();
print(df_concat4);
```

### Concatenations with joins

When concatinating dataframes with different column names `pd.concat` defaults to filling entries where no data is available with NA values.

```{code-cell} ipython3
df5 = create_df(["red", "green", "cyan"], [1, 2, 3])
df6 = create_df(["green", "cyan", "blue"], [4, 5, 6])
print(df5); print()
print(df6); print()
print(pd.concat([df5, df6]))
```

In order to remove the NA's we want to use the argument `join` to concatenate the function. The default statement is a union join, `join = 'outer'`. This can be changed to an intersection of the columns by using `join = 'inner'`.

```{code-cell} ipython3
print(df5); print();
print(df6); print();
print(pd.concat([df5, df6], join = 'inner'))
```

### Append() Method

Direct array concatentation is very common, `series` and `DataFrame` objects have an additional method that can be used in similar fashion.

```{code-cell} ipython3
print(df1); print();
print(df2); print();
print(df1.append(df2))
```

`.append()` is a very simple function that does not check for duplicate indicies and will create a new DataFrame that is simply both DataFrames together, as seen here.

```{code-cell} ipython3
print(df1); print();
print(df1); print();
print(df1.append(df1))
```

## Combining Datasets: Merge and Join

+++

### Categories of Joins

Using the `pd.merge()` function carries out a number of types of joins: *one-to-one*, *many-to-one*, and *many-to-many* joins.

+++

#### One-to-One Join
A one-to-one join is perhaps the simplist, being very simmilar to column-wise concatenation. To do this we simply run `pd.merge()` with our data frames as arguments. In this case `pd.merge()` sees that the `ID` column is shared between df7 and df8 and will use it as a key to merge the two. What results is an intersection of the two DataFrames.

```{code-cell} ipython3
df7 = pd.DataFrame({
    'ID': [101, 102, 103, 104, 105, 106, 107],
    'Product_Name': ['SmartWatch', 'Backpack', 'Shoes', 'Smartphone',
                     'Books', 'Oil','Laptop'],
    'Category': ['Electronics', 'Study', 'Fashion', 'Electronics',
                 'Study', 'Grocery', 'Electronics'],
    'Price': [299.0, 150.50, 2999.0, 14999.0, 145.0, 110.0, 79999.0]
})

df8 = pd.DataFrame({
    'Ref_Num': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Name': ['Ollie', 'Ivy', 'Ethan', 'Maya', 
             'Lucas', 'Levi', 'Miles', 'Daniel', 
             'Owen'],
    'Age': [20, 25, 15, 10, 30, 65, 35, 18, 23],
    'ID': [101, 0, 106, 0, 103, 104, 0, 0, 107],
    'Purchased_Product': ['SmartWatch', 'NA', 'Oil', 'NA', 'Shoes', 
                          'Smartphone','NA','NA','Laptop']
})

df7 # Left DataFrame
```

```{code-cell} ipython3
df8 # Right DataFrame
```

```{code-cell} ipython3
# Using pd.merge()
df9 = pd.merge(df7, df8)
df9
```

#### Many-to-One Joins

Many-to-one joins are joins in which one of the two key columns contains duplicated entries. This is simmilarly done by calling `pd.merge()` and will result in the new DataFrame preserving the duplicated entries as appropriate. In this example the `Category` columns are shared and used as a key to merge the two DataFrames. In essence a new column was added to df9 called `Need` using df10 almost like a dictionary to determine what goes in `Need` for each entry.

```{code-cell} ipython3
df10 = pd.DataFrame({
    'Category': ['Electronics', 'Fashion', 'Grocery'],
    'Need': ['Personal', 'Unnecessary', 'Necessary']
})

df9 # Left Dataframe
```

```{code-cell} ipython3
df10 # Right DataFrame
```

```{code-cell} ipython3
# Using pd.merge()
pd.merge(df9, df10)
```

#### Many-to-Many Joins

Many-to-many joins if the key column in both left and right DataFrame contains any duplicates, it can result in many-to-many joins. Calling `pd.merge()` will preserve the duplicate entries for both. In this example the key column is `Category` and for both DataFrames there are some duplicate entries. It preserves these by creating new instances for each combination possible. So for here there are 3 entries for Electronics for df7 and 2 entries for Electronics for df11. Combining those there is now 6 entries for Electronics, representing the 6 possible combinations.

```{code-cell} ipython3
df11 = pd.DataFrame({
    'Category': ['Electronics', 'Electronics', 'Study', 'Study',
                 'Fashion', 'Fashion', 'Grocery', 'Grocery'],
    'Occupation': ['College', 'Part-Time', 'Retired', 'Not Working',
                   'Working', 'Full-Time', 'High School', 'Chores']
})

df7 # Left DataFrame
```

```{code-cell} ipython3
df11 # Right DataFrame
```

```{code-cell} ipython3
# Using pd.merge()
pd.merge(df7, df11)
```

### Using Merge Keys

By default, `pd.merge()` looks for one or more matching column names to use as a key. However you can specify this column name using the argument `on = 'column_name'`.

```{code-cell} ipython3
print(df7); print()
print(df11); print()

# using on 
print(pd.merge(df7, df11, on = 'Category')); print()
```

If two colums that are the same have different names you can merge them using `left_on` and `right_on` to specify which column for each DataFrame is supposed to be the key. This will also preserve both columns in the resulting DataFrame, however they will be identical besides the name.

```{code-cell} ipython3
df12 = pd.DataFrame({
    'Type': ['Electronics', 'Electronics', 'Study', 'Study',
                 'Fashion', 'Fashion', 'Grocery', 'Grocery'],
    'Occupation': ['College', 'Part-Time', 'Retired', 'Not Working',
                   'Working', 'Full-Time', 'High School', 'Chores']
})

print(df7); print()
print(df12); print()

# using left_on and right_on
print(pd.merge(df7, df12, left_on = 'Category', right_on = 'Type'))
```

It is even possible to merge by index using the arguments `left_index = True` and `right_index = True` to specify you want to merge by index.

```{code-cell} ipython3
df13 = pd.DataFrame({
    'Name': ['Kelly', 'Ann', 'Suzy', 'Bob',
            'Sydney', 'Jacob', 'Will'],
    'Occupation': ['College', 'Part-Time', 'Retired', 'Not Working',
                   'Working', 'Full-Time', 'High School']
})

df14 = pd.DataFrame({
    'Name': ['Jacob', 'Sydney', 'Suzy', 'Will',
             'Ann', 'Kelly', 'Bob'],
    'Favorite_Color': ['Blue', 'Red', 'Green', 'Yellow',
                       'Orange', 'Brown', 'Purple']
})

df13 = df13.set_index('Name')
df14 = df14.set_index('Name')
print(df13); print()
print(df14); print()

# Using left_index and right_index
print(pd.merge(df13, df14, left_index = True, right_index = True))
```

## Aggregation and Grouping

+++

### Simple Aggregration 

For Pandas `DataFrame` aggrigates return results within each column. All common aggrigates are available, and in addition there is a method `describe()` which computes several common aggrigates at once for each column.

```{code-cell} ipython3
#!pip3 install seaborn
import seaborn as sns
mpg = sns.load_dataset('mpg')

mpg.dropna().describe()
```

### GroupBy

Conditional Aggrigation by some label or index can be done by `groupby` operation, which does the "split, apply, combine" operation by default.

```{code-cell} ipython3
mpg.dropna().groupby('origin').mean()
```

### Using GroupBy Object

It is possible to think of the `GroupBy` object as a collection of `DataFrames`, and it has a variety of operations that can be used. It is possible to index a `GroupBy` object as you would a `DataFrame` to return a modified GroupBy object.

```{code-cell} ipython3
# Index the mpg data grouped by origin to look at the median
mpg.groupby('origin')['mpg'].median()
```

The `GroupBy` object also supports direct iteration over groups, returning each group as a `Series` or `DataFrame`.

```{code-cell} ipython3
# Iterate over the origin 
for (origin, group) in mpg.groupby('origin'):
    print("{0:30s} shape = {1}".format(origin, group.shape))
```

In addition any method not specifically called by the `GroupBy` object will be called on the indivdual groups within the `GroupBy` object.

```{code-cell} ipython3
# Applying the describe() method to each group after
# grouping by region of origin
mpg.groupby('origin')['mpg'].describe()
```

### GroupBy Aggregration 

The aggregate function can take a string or function or list of those and compute all aggregates at once.

```{code-cell} ipython3
df17 = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', "Blue"],
    'Data1': range(6),
    'Data2': np.random.randint(0, 10, 6)},
    columns = ['Color', 'Data1', 'Data2']
)

# Returns the groups by color looking at the aggregates 
df17.groupby('Color').aggregate(['min', np.median, max])
```

You can also pass a dictionary which maps colum names to operations to be used in those columns.

```{code-cell} ipython3
# Returns only the min for data1 and max for data2
df17.groupby('Color').aggregate({'Data1':'min',
                                 'Data2':'max'})
```

### GroupBy Filtering 

Filtering allows us to keep data based on group properties. It returns a Boolian value saying if the group passes the filtering.

```{code-cell} ipython3
# Keep data only if standard deviation is greater than 3
def filter_function(x):
    return x['Data2'].std() > 4

print(df17); print()
print(df17.groupby('Color').std()); print()
print(df17.groupby('Color').filter(filter_function))
```

### GroupBy apply() Method

`apply()` lets you apply an arbitrary function to the results of a group. The function takes a `DataFrame` as an argument and returns either a Pandas `DataFrame`, Pandas `Series` or a scalar.

```{code-cell} ipython3
# Divides data1 by the mean of data2
def dev_by_mean_data2(x):
    x['Data1'] /= x['Data2'].mean()
    return(x)

print(df17); print()
print(df17.groupby('Color').apply(dev_by_mean_data2))
```

### Specifying the Split Key

The `DataFrame` can be split by more than just a single column name. It can be split by any list, array, series, or index providing the grouping keys so long as the length matches the `DataFrame`.

```{code-cell} ipython3
# Split using a list of groups in order for each 
# index in the dataframe
L = [1, 0, 1, 2, 3, 1]
print(df17); print()
print(df17.groupby(L).sum()); print()
```

It can also be split by a dictionary which maps index values to group keys.

```{code-cell} ipython3
# Split using a dictionary mapping colors to if 
# they are primary or secondary
df18 = df17.set_index('Color')
mapping = {'Red':'Primary', 'Blue':'Primary', 'Green':'Secondary'}
print(df18); print()
print(df18.groupby(mapping).sum()); print()
```

It can also be split by any Python function so long as it inputs the index value and outputs the group.

```{code-cell} ipython3
# Split using str.upper to make all indicies uppercase
print(df18); print()
print(df18.groupby(str.upper).mean()); print()
```

Finally it can be done by mixing any of these together in a list of valid keys to create a multi-index.

```{code-cell} ipython3
# Split using both str.upper and the dictionary mapping 
# colors to primary and secondary
print(df18); print()
print(df18.groupby([str.upper, mapping]).mean())
```

## Pivot Tables

+++

### Basics

While `groupby` is useful for gaining basic understanding of data, it can become messy when you try to do anything in more than one-dimension. This is why Pandas has the built in routine `pivot_table` which can easily handle multidimensional aggregation. `pivot_table` allows us to generate a new table of aggrigates which can be broken down further to allow for multidimensional analysis. For example here we are finding the mean mpg for cars based on region of origin and their number of cylinders at the same time.

```{code-cell} ipython3
mpg.pivot_table('mpg', index = 'origin', columns = 'cylinders')
```

### Multilevel Pivot Tables

We can bin data to show multilevel tables using the `pd.cut` and `pd.qcut` functions. Here are examples of three-dimensional and four-dimensional tables done in this way.

```{code-cell} ipython3
# Three-dimensional table looking at model year as a third dimension
years = pd.cut(mpg['model_year'], [70, 73, 76, 79, 82])
mpg.pivot_table('mpg', ['origin', years], 'cylinders')
```

```{code-cell} ipython3
# Four-dimensional table dividing the results into two weight categories
weight = pd.qcut(mpg['weight'], 2)
mpg.pivot_table('mpg', ['origin', years], ['cylinders', weight])
```

### Additional Pivot Tables Options

There are five arguments we havent covered for pivot tables. `fill_value` and `dropna` deal with missing data. `aggfunc` keyword determines which type of aggrigation is applied, which is mean by default. It can specify `'sum'`, `'mean'`, `'count'`, `'min'`, `'max'`, etc.. or a function for an aggrigation (`np.sum()`, `min()`, `sum()`, etc) It can also be a dictionary mapping a collumn to any of the previous options. When using aggfunc the values keyword is determined automatically

```{code-cell} ipython3
# Pivot table looking at both the median mpg and max 
# horsepower for each cylinder count for each region
mpg.pivot_table(index = 'origin', columns = 'cylinders', 
                aggfunc = {'mpg':'median', 'horsepower':'max'})
```

The `margins` keyword can be used to compute totals along each grouping. The name defaults to "All" but can be specified using `margins_names = "your_name"`

```{code-cell} ipython3
# Pivot tqble looking at mean mpg per region and in all regions
mpg.pivot_table('mpg', index = 'origin', columns = 'cylinders',
                margins = True)
```
