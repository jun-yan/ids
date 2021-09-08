#!/usr/bin/env python
# coding: utf-8

# <!-- (about_py)= -->
# 
# # Python for R Users
# 
# [Workshop by Mango
# Solutions](https://github.com/MangoTheCat/python-for-r-users-workshop)
# 
# - Jupyter notebook keyboard shortcuts
# - Models, packages, and libraries
# - Getting help
# 
# ## Reading data and manipulations

# In[1]:


import pandas as pd

mtcars = pd.read_csv("../data/mtcars.csv", index_col = 0)
mtcars.head(5)
mtcars[['mpg', 'wt']]


# ## Writing a function
# 
# Consider the Fibonacci Sequence
# $1, 1, 2, 3, 5, 8, 13, 21, 34, ...$.
# The next number is found by adding up the two numbers before it.
# We are going to use 3 ways to solve the problems.
# 
# 1. Recurisive solution;

# In[2]:


def fib_rs(n):
    if (n==1 or n==2):
        return 1
    else:
        return fib_rs(n - 1) + fib_rs(n - 2)

import timeit
start = timeit.default_timer()
print(fib_rs(35))
stop = timeit.default_timer()
print('Time: ', stop - start)


# 1. Dynamic programming memoization;

# In[3]:


def fib_dm_helper(n,mem):
    if mem[n] is not None:
        return mem[n]
    elif (n==1 or n==2):
        result = 1
    else:
        result = fib_dm_helper(n-1,mem)+fib_dm_helper(n-2,mem)
    mem[n]=result
    return result

def fib_dm(n):
    mem=[None]*(n+1)
    return fib_dm_helper(n,mem)
	
start = timeit.default_timer()
print(fib_dm(35))
stop = timeit.default_timer()
print('Time: ', stop - start) 


# 1. Dynamic programming bottom-up.

# In[4]:


def fib_dbu(n):
    mem=[None]*(n+1)
    mem[1]=1;
    mem[2]=1;
    for i in range(3,n+1):
        mem[i]=mem[i-1]+mem[i-2]
    return mem[n]
	

start = timeit.default_timer()
print(fib_dbu(500))
stop = timeit.default_timer()
print('Time: ', stop - start) 


# # General Resources
# Popular textbooks on Python programming include {cite}`guttag2016introduction` and {cite}`hill2016learning`.
# 
# 
# Python is either the dominant player or a major player in
# 
# -   [machine learning and data science](http://scikit-learn.org/stable/)
# -   [astronomy](http://www.astropy.org/)
# -   [artificial intelligence](https://wiki.python.org/moin/PythonForArtificialIntelligence)
# -   [chemistry](http://chemlab.github.io/chemlab/)
# -   [computational biology](http://biopython.org/wiki/Main_Page)
# -   [meteorology](https://pypi.org/project/meteorology/)
# 
# 
# ## Bibliography
# 
# ```{bibliography} ../_bibliography/references.bib
# ```
