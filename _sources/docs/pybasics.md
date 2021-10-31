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
pd.set_option("display.width", 80)

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

```{code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", data = mtcars)
sns.displot(mtcars, x = "mpg", col = "gear", binwidth = 3, height = 3)
```

```{code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", hue = "gear", data = mtcars)
```

```{code-cell} ipython3
sns.lmplot(x = "mpg", y = "wt", col = "gear", data = mtcars)
```

## Numbers in Computer

Pitfall one: Integer overflow

We my encounter the Overflow problem when dealing with integers when using Numpy and Pandas in Python.

```{code-cell} ipython3
import numpy as np
x = np.array(2**63 -1 , dtype='int')
x
# This should be the largest number numpy can display, with the default integer size being 64 bits in most computers.

y = np.array(x + 1, dtype='int')
y
# Because of the overflow, when we add 1 to this number, it becomes negative!
```

For vanilla Python, the overflow errors are checked and more digits are allocated when needed, at the cost of being slow.

```{code-cell} ipython3
2**63 * 1000 # this number is 1000 times largger than the prior number, but still displayed perfectly without any overflows
```

Pitfall two: floating point numbers' impresicion

Because our computers use binary to store information, some simple numbers in decimal are not represented and replaced by a rounded approximation.

```{code-cell} ipython3
0.1 + 0.1 + 0.1 == 0.3

0.3 - 0.2

import decimal
decimal.Decimal(0.1) # this is the true decimal value of the binary approximation stored for 0.1
```

Pitfall three: trade-off between size and presicion

To represent a floating points number, a 64-bit computer typically uses 1 bit to strore the sign, 52 bits to store the mantissa and 11 bits to store the exponent. Because the mantissa bits are limited, it can not represent a floating point that's both very big and very precise. Most computers can represent all integers up to 2^53, after that it starts skipping numbers.

```{code-cell} ipython3
2.1**53 +1 == 2.1**53

# Find a number larger than 2 to the 53rd
```

```{code-cell} ipython3
x = 2.1**53
for i in range(1000000):
    x = x+1
x == 2.1**53

# we add 1 to x by 1000000 times, but it still equal to its initial value, 2.1**53, because this number is too big that computer can't handle it with precision like add 1.
```

For floating points, we can find the smallest value we can get by calling machine epsilon. It is formally defined as the difference between 1 and the next largest floating point number.

```{code-cell} ipython3
float_epsilon = np.finfo(float).eps
print(float_epsilon)
```

## Example: Google recruiting problem
Find the first 10-digit prime found in consecutive digits of $e$.

- Get the list of digits of $e$
- Write functions to achieve our goals(can be a loop, running over the list)
- Try to make it more efficient!

### Export $e$

If you try to export $e$ using the format command in Python, you will
not get the precise $e$ value due to the double type data's feature!

```{code-cell} ipython3
import math
print("%0.10f" % math.e)
print("%0.20f" % math.e)
```

The digits after 16 positions of $e$ are wrong, compare this to the
$e$ value export by "decimal".

```{code-cell} ipython3
import operator # ?
import decimal  # for what?

## set the required digits
decimal.getcontext().prec = 150
e_decimal = decimal.Decimal(1).exp().to_eng_string()[2:]
print(e_decimal)
```

Besides the "decimal" module, we have lazy ways to find $e$ from
an existed list. For example, the website
https://apod.nasa.gov/htmltest/gifcity/e.2mil provides a list of $e$.

```{code-cell} ipython3
import requests
# get text from the website
reply = requests.get('https://apod.nasa.gov/htmltest/gifcity/e.2mil').text

# remove the space in the lines
line_strip=[line.strip() for line in reply.split('\n')]

# connect all the digital lines
e=''.join([LINE for LINE in line_strip if LINE and LINE[0].isdigit()])
print(e[:20:])
```

### Write functions to check whether a number is a prime
Here the most basic and bruteforce way is used. Check all the factors
less than $\sqrt n+1$ for all the odd numbers $n$.

We have many powerful algorithms to achieve this if we need to check
for very large numbers. For this problem, this method is efficcient
enough.

```{code-cell} ipython3
%%time
def is_prime(n):
    if n% 2 == 0:
        return False
    for i in range(3, int(n**0.5)+1,2):
        if n% i == 0:
            return False
    else:
        return True
    

i = 0
se = e.replace('.','')
while True:
    number = se[i:i+10]
    if is_prime(int(number)):
        print(number)
        break
    i+= 1
```

## Error and Debugging
Resource: Python Data Science Handbook

Code development and data analysis always require a bit of trial and error, and IPython contains tools to streamline this process. 

### Error

```{code-cell} ipython3
def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x + 1
    return func1(a, b)
```

```{code-cell} ipython3
try:
    func2(-1)
except Exception as e:
    print(e)
```

Calling func2 results in an error, and reading the printed trace lets us see exactly what happened.By default, this trace includes several lines showing the context of each step that led to the error.

### Debugging

The standard Python tool for interactive debugging is pdb, the Python debugger. This debugger lets the user step through the code line by line in order to see what might be causing a more difficult error. The IPython-enhanced version of this is ipdb, the IPython debugger.

```{code-cell} ipython3
%debug
```

The interactive debugger allows much more than this, though–we can even step up and down through the stack and explore the values of variables there

### Partial list of debugging commands
There are many more available commands for interactive debugging than we've listed here; the following table contains a description of some of the more common and useful ones:

**Command	Description**

**list**	Show the current location in the file  
**h(elp)**	Show a list of commands, or find help on a specific command  
**q(uit)**	Quit the debugger and the program  
**c(ontinue)**	Quit the debugger, continue in the program  
**n(ext)**	Go to the next step of the program  
**(enter)**	Repeat the previous command  
**p(rint)**	Print variables  
**s(tep)**	Step into a subroutine  
**r(eturn)**	Return out of a subroutine

+++

## General Resources
Popular textbooks on Python programming include
{cite}`guttag2016introduction` and {cite}`hill2016learning`.



Python is either the dominant player or a major player in

-   [machine learning and data science](http://scikit-learn.org/stable/)
-   [astronomy](http://www.astropy.org/)
-   [artificial intelligence](https://wiki.python.org/moin/PythonForArtificialIntelligence)
-   [chemistry](http://chemlab.github.io/chemlab/)
-   [computational biology](http://biopython.org/wiki/Main_Page)
-   [meteorology](https://pypi.org/project/meteorology/)


## Bibliography

```{bibliography} ../_bibliography/references.bib
```

## Exercises

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

1. Write a function to obtain the MLE of the parameters of a gamma
   distribution with a random sample. The input is the random sample
   vector, and the output is the MLE. For shape parameter $\alpha$ and
   scale parameter $\beta$, generate a random sample of size $n$, and
   then use your function to obtain the MLE. Conduct a simulation
   study with $\alpha = \beta = 2$ for sample size 
   $n \in \{50, 100, 200\}$. Do 1000 replicates and summarize your
   results.

1. Continue with the gamma MLE. Add the standard error of the MLE into
   the output of your function. Repeat the simulation study and check
   whether your standard errors match the empirical standard error
   from the 1000 replicates.

1. Consider a telecommunication service company. To simplify the
   setting, suppose that the starting points of new customers are
   uniformly distributed over time. Suppose that a customer stays
   with the company for a duration of gamma distribution with shape
   `alpha` and scale `sigma`. The company hires a data scientist to
   estimate the distribution of the duration of the customer's
   stay. The data scientist decides to include in the sample all
   customers who are with the company at time `t`. Write a function to
   conduct a simulation study. For each replicate, the function
   generates `n` customers and select those that are active at time
   `t` to form a sample; returns the mean and standard deviation of
   the durations in the sample. Repeat the experiment 1000 times for
   various values of `alpha`, `sigma`, `n`, and `t`. Discuss
   your findings.

1. Continue with the telecommunication service company. Suppose now
   that the data scientists decides to include in the sample all
   custermers that are active during a time interval from `t0` to
   `t1`. Modify your function and carry out the simulation
   study. Discuss your findings.
