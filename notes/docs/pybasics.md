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

# Python for R Users

[Workshop by Mango
Solutions](https://github.com/MangoTheCat/python-for-r-users-workshop)

- Jupyter notebook keyboard shortcuts
- Models, packages, and libraries
- Getting help

## Reading data and manipulations

```{code-cell} ipython3
import pandas as pd

mtcars = pd.read_csv("../data/mtcars.csv", index_col = 0)
mtcars.head(5)
mtcars[['mpg', 'wt']]
```

## Writing a function

Consider the Fibonacci Sequence
$1, 1, 2, 3, 5, 8, 13, 21, 34, ...$.
The next number is found by adding up the two numbers before it.
We are going to use 3 ways to solve the problems.

1. Recurisive solution;

```{code-cell} ipython3
def fib_rs(n):
    if (n==1 or n==2):
        return 1
    else:
        return fib_rs(n - 1) + fib_rs(n - 2)

%timeit fib_rs(10)
```

1. Dynamic programming memoization;

```{code-cell} ipython3
def fib_dm_helper(n,mem):
    if mem[n] is not None:
        return mem[n]
    elif (n==1 or n==2):
        result = 1
    else:
        result = fib_dm_helper(n - 1, mem) + fib_dm_helper(n - 2, mem)
    mem[n]=result
    return result

def fib_dm(n):
    mem=[None]*(n+1)
    return fib_dm_helper(n, mem)

%timeit fib_dm(10)
```

1. Dynamic programming bottom-up.

```{code-cell} ipython3
def fib_dbu(n):
    mem=[None]*(n+1)
    mem[1]=1;
    mem[2]=1;
    for i in range(3,n+1):
        mem[i] = mem[i-1] + mem[i-2]
    return mem[n]
	

%timeit fib_dbu(500)
```

## Variables versus Objects
See materials and exercises of Dr. Eubank's [PDS site](https://www.practicaldatascience.org/html/index.html).


## Visualization
Change the default plot format to `svg` for high quality display.

```{code-cell} ipython3
%config InlineBackend.figure_formats = ['svg']

import seaborn as sns
%matplotlib inline

sns.set_theme(style="darkgrid")
df = sns.load_dataset("penguins")
fig = sns.displot(
    df, x="flipper_length_mm", col="species", row="sex",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True)
)
```

Let's see some plots using the `mtcars` example.
``` {code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", data = mtcars)
sns.displot(mtcars, x = "mpg", col = "gear", binwidth = 3, height = 3)
```

``` {code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", hue = "gear", data = mtcars)
```

``` {code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", col = "gear", data = mtcars)
```


# General Resources
Popular textbooks on Python programming include {cite}`guttag2016introduction` and {cite}`hill2016learning`.


Python is either the dominant player or a major player in

-   [machine learning and data science](http://scikit-learn.org/stable/)
-   [astronomy](http://www.astropy.org/)
-   [artificial intelligence](https://wiki.python.org/moin/PythonForArtificialIntelligence)
-   [chemistry](http://chemlab.github.io/chemlab/)
-   [computational biology](http://biopython.org/wiki/Main_Page)
-   [meteorology](https://pypi.org/project/meteorology/)


# Bibliography

```{bibliography} ../_bibliography/references.bib
```

# Exercises

1. Write a function to demonstrate the Monty Hall problem through
   simulation. The function takes two arguments `ndoors` and
   `ntrials`, representing the number of doors in the experiment and
   the number of trails in a simulation, respectively. The function
   should return the proportion of wins for both the switch and
   no-switch strategy. Apply your function with 3 doors and 5 doors,
   both with 1000 trials.

1. Write a function to do a Monte Carlo approximation of $\pi$. The
   function takes a Monte Carlo sample size `n` as input, and returns
   a point estimate of $\pi$ and a 95% confidence interval. Apply your
   function with sample size 1000, 2000, 4000, and 8000. Comment on
   the results.

1. Find the first 10-digit prime number occurring in consecutive
   digits of $e$. This was a
   [Google recruiting ad](http://mathworld.wolfram.com/news/2004-10-13/google/).
