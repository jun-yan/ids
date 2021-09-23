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
except:
    print("division by zero")
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


## Profiling


### Motivation for Profiling
Speed? Jump the bottlenecks
Which one do you want your code to be?

!["https://www.seekpng.com/png/detail/127-1274961_visit-zootopia-jpg-free-sloth-characters-in-zootopia.png"](Slow)

!["https://freepngimg.com/thumb/the_flash/5-2-flash-free-png-image.png"](Fast)


This could be one most common reason for timing and profiling. The understanding
is that only some small chunk often makes our code heavy in terms of time taken,
memory usage, etc. Finding it and improving or managing it can improve our
resource usage.

It is especially important to profile the code chunks to optimize. This becomes
more important in a production environment where resources are expensive and
cannot be wasted. Therefore, as part of testing, one should include profiling to
see which chunks use most resources and which chunks use less. Some coding
contests often have a maximum time to run or people boast of how fast their
particular solution runs

### Resources mostly profiled

The most common resources that we often optimize for projects are: 

- CPU time
- Memory Usage

But some other resources could also be network bandwidth usage etc. For example,
if you wish to create a notebook that gathers stock data from one of many
locations such as Yahoo Finance etc, and you only have a limited bandwidth, your
choice could also depend on where you gather your data from.

### Profiling CPU time
Profiling implies that it divides the entire code file into smaller code chunks
(*functions/lines*) in order to show the resource usage and thus existence of
bottlenecks in our code. It is best practice to find bottlenecks by means of
profiling rather than by intuition.

Python includes the following modules that achieve profiling for CPU time:

- timeit
- cprofile
- profile
- lineprofile

For interpretation, we use the fibonacci example shown in class:

**Note: The profiler modules are designed to provide an execution profile for a
given program, not for benchmarking purposes (for that, there is timeit for
reasonably accurate results). This particularly applies to benchmarking Python
code against C code: the profilers introduce overhead for Python code, but not
for C-level functions, and so the C code would seem faster than any Python
one.**

### `timeit` Function Usage:


```python
import pandas    ## !pip3 install pandas (uncomment to add)
import timeit    ## base package

### First interpretation using dynamic programming memorization
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

## One  liners can be done using timeit magic function 
## %timeit (this is a feature of ipython)
%timeit fib_dm(100)
```


```python
### Second Interpretation using dynamic programming bottom up
def fib_dbu(n):
    mem=[None]*(n+1)
    mem[1]=1;
    mem[2]=1;
    for i in range(3,n+1):
        mem[i] = mem[i-1] + mem[i-2]
    return mem[n]
	

fib_dbu(100)
```


```python
%timeit fib_dbu(100)  ## timeit application of fib_dbu
```

### Dynamic save of a line to a code file

This is a great tool (in ipython) as you can work on some code functions and
applications and then save each chunk correspondingly to the "source" file with
functions and "application" file once you ensure it works on the notebook. It
helps a great deal as you'd like your functions to finally appear in a ".py"
file to automate it in clusters or so on.


```python
!cp myscript.py myscript2.py
%save -a myscript2.py -n 4   ##(-a for append, -n for the line run 
## -n may change every session, thus good for mostly dynamic usage))
```

### Command Line Interface for timeit

```python
!python -m timeit -r 20 '"-".join(str(n) for n in range(100))'  ## r does repetition
## The output suggests there were 20000 loops, repeated 20 times for accuracy, and 

"-".join(str(n) for n in range(100))
```

### cProfile for profiling CPU time

Source: [The python Profilers](https://docs.python.org/3/library/profile.html)

To profile a function that takes only one argument, you can do the following:


```python
import cProfile   ## for profiling function
```

```python
# cProfile.run('re.compile("foo|bar")')
cProfile.run('fib_dm(2000)')
cProfile.run('fib_dbu(2000)')
```

- The first line indicates that 21 calls were monitored of which 5 were
  primitive calls (calls that are not induced via recursion).
- "Ordered by: standard name" suggests usage of the last column for sorting the
  output.
- When there are two numbers in the first column (for example 3/1), it means
  that the function recursed.
- (x/y) for the first column says that y is the primitive calls and x is the
  recursive calls.
- You can also save the run output to a file by providing it as the second
  argument as follows:


```python
cProfile.run('fib_dm(2000)', 'restats')
```

### Command Line for cProfile

The files cProfile and profile can also be invoked as a script to profile
another script. For example:


```python
## run using bash and saved `py` file
!python -m cProfile -o restats myscript.py   ## profiling an entire script will include
                                             ## overhead time for importing, creating a function
## Once again run at the end
cProfile.run('fib_dm(2000)', 'restats')
```

### Interpretation of Raw Output format:

The column headings include:
- ncalls: for the number of calls.
- tottime: for the total time spent in the given function (and excluding time
  made in calls to sub-functions)
- percall: is the quotient of tottime divided by ncalls
- cumtime: is the cumulative time spent in this and all subfunctions (from
  invocation till exit). This figure is accurate even for recursive functions.
- percall: is the quotient of cumtime divided by primitive calls
- filename: lineno(function) provides the respective data of each function

The pstats.Stats class reads profile results from a file and formats them in
readable manner.


```python
# !pip install pstats   # pip install from within notebook using bash magic command
import pstats  ## Clean reprsentation of profile
from pstats import SortKey
p = pstats.Stats('restats') ## uses the above file created with output
;
```


```python
### some possible print outputs
p.strip_dirs().sort_stats(-1).print_stats(.1)
p.sort_stats(SortKey.CUMULATIVE).print_stats(10) ## sort cumulative time spent
p.sort_stats(SortKey.TIME).print_stats(10)  ## sort time spent within each function
```

### Deterministic Profiling of cProfile
Deterministic profiling is meant to reflect the fact that all function call,
function return, and exception events are monitored, and precise timings are
made for the intervals between these events (during which time the user’s code
is executing). In contrast, statistical profiling (which is not done by cprofile
module) randomly samples the effective instruction pointer, and deduces where
time is being spent. The latter technique traditionally involves less overhead
(as the code does not need to be instrumented), but provides only relative
indications of where time is being spent.


### Interpretation of cProfile
Call count statistics can be used to identify bugs in code (surprising counts),
and to identify possible inline-expansion points (high call counts). Internal
time statistics can be used to identify “hot loops” that should be carefully
optimized. Cumulative time statistics should be used to identify high level
errors in the selection of algorithms. Note that the unusual handling of
cumulative times in this profiler allows statistics for recursive
implementations of algorithms to be directly compared to iterative
implementations.

### Line Profiler

*line_profiler will profile the time individual lines of code take to
execute. The profiler is implemented in C via Cython in order to reduce the
overhead of profiling.* Also the timer unit is $10^{-6}$ or $\mu s$


```python
!pip3 install line_profiler  ## install line_profiler
```


```python
from line_profiler import LineProfiler
import random

def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]    ## Notice different for loop
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

numbers = [random.randint(1,100) for i in range(1000)]
lp = LineProfiler()
lp_wrapper = lp(do_stuff)
lp_wrapper(numbers)
lp.print_stats()
```


```python
## from pycallgraph import PyCallGraph
## from pycallgraph.output import GraphvizOutput

## with PyCallGraph(output=GraphvizOutput()):
##    fib_dm(10)
!gprof2dot -f pstats restats | dot -Tsvg -o mine.svg
```


```python
from IPython.display import SVG
SVG('mine.svg')
```

### Profiling Memory

- memory_profiler open source  python package
- guppy

### Memory_Profiler

```python
!pip3 install -U memory_profiler    ## install memory_profiler
```

```python
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

```python
%save -a example.py -n 28   ## change 28 to respective line number
```

```python
!python3.9 -m memory_profiler example.py   ## calc. memory used by my function 

## This goes into the pdb debugger as soon as 100MB is used
## !python3.9 -m memory_profiler --pdb-mmem=100 example.py  
```

```python
## This can be improvised by mprof
!pip3 install matplotlib
import matplotlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
```


```python
!mprof run example.py
!mprof plot -o image.jpeg
```

!["Mprof Output"](image.jpeg)

More information about this and even plots etc can be found at
[Memory_Profiler](https://pypi.org/project/memory-profiler/). You can use a
memory profiling by putting the @profile decorator around any function or method
and running python -m memory_profiler myscript. You'll see line-by-line memory
usage once your script exits.

### Guppy

There are number of ways to profile an entire Python application. Most python
way is Guppy. You can take a snapshot of the heap before and after a critical
process. Then compare the total memory and pinpoint possible memory spikes
involved within common objects. Well, but apparently it works with Python 2.


```python
# !pip install guppy

# from guppy import hpy
# h = hpy()
# h.heap()
    
# !python3.9 example.py

# from guppy import hpy
# h = hpy()
# h.heap()

## multiline commenting is CMD + /
```

Objgraph also helps in finding memory leaks using visualization as follows:


```python
!pip3 install objgraph
```


```python
import objgraph
objgraph.show_most_common_types()   ## overview of the objects in memory
```


```python
class MyBigFatObject(object):
    pass

def computate_something(_cache={}):
    _cache[42] = dict(foo=MyBigFatObject(),
                      bar=MyBigFatObject())
    # a very explicit and easy-to-find "leak" but oh well
    x = MyBigFatObject() # this one doesn't leak
```


```python
objgraph.show_growth(limit=3) 
computate_something()
objgraph.show_growth() 
```

It’s easy to see MyBigFatObject instances that appeared and were not freed. So,
we can trace the reference chain back.

+++


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

