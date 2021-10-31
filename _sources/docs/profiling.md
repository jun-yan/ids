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

# Profiling

## Motivation for Profiling

Time optimization is one most common reason for timing and
profiling. The understanding is that only some small chunk often makes
our code heavy in terms of time taken, memory usage, etc. Finding it
and improving or managing it can improve our resource usage.


|         `Slow`           |           `Fast`         |
:-------------------------:|:-------------------------:
<img src="https://www.seekpng.com/png/detail/127-1274961_visit-zootopia-jpg-free-sloth-characters-in-zootopia.png" width="425"/> | <img src="https://freepngimg.com/thumb/the_flash/5-2-flash-free-png-image.png" width="425"/> 


It is especially important to profile code chunks to optimize them. This becomes
more important in a production environment where resources are expensive and
cannot be wasted. Therefore, as part of testing, one should include profiling to
see which chunks use most resources and which chunks use less. Some coding
contests often have a maximum time to run and these situations also
lead to its necessity.

## Resources mostly profiled

The most common resources that we often optimize for projects are: 

- CPU time
- Memory Usage

But some other resources could also be network bandwidth usage etc. For example,
if you wish to create a notebook that gathers stock data from one of many
locations such as Yahoo Finance etc, and you only have a limited bandwidth, your
choice could also depend on where you gather your data from.

## Profiling CPU time
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

````{note}
The profiler modules are designed to provide an execution profile for a
given program, not for benchmarking purposes (for that, there is timeit for
reasonably accurate results). This particularly applies to benchmarking Python
code against C code: the profilers introduce overhead for Python code, but not
for C-level functions, and so the C code would seem faster than any Python
one.
````

### `timeit` function usage:

```{code-cell} ipython3
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

%timeit fib_dm(100)
```

```{code-cell} ipython3
### Using dynamic programming bottom up for timeit
def fib_dbu(n):
    mem=[None]*(n+1)
    mem[1]=1;
    mem[2]=1;
    for i in range(3,n+1):
        mem[i] = mem[i-1] + mem[i-2]
    return mem[n]
	

%timeit fib_dbu(100)  ## timeit application of fib_dbu
```

#### Command Line Interface for `timeit`

```{code-cell} ipython3
!python -m timeit -r 20 '"-".join(str(n) for n in range(100))'  ## r does repetition
## The output suggests there were 20000 loops, repeated 20 times for accuracy, and 
```

### `cProfile`  function usage

`cProfile` allows to profile functions' CPU time. A lot more info on how this can be used can also be obtained at Source: [[The python Profilers]](https://docs.python.org/3/library/profile.html).  

For profiling a function with one argument, you can do the following:

```{code-cell} ipython3
import cProfile   ## for profiling function
```

```{code-cell} ipython3
cProfile.run('fib_dm(2000)')
cProfile.run('fib_dbu(2000)')
```

**Interpretation of the above `cProfile.run()`**:

- The first line indicates that 4001 calls were monitored of which 5 were
  primitive calls (calls that are not induced via recursion).
- "Ordered by: standard name" suggests usage of the last column for sorting the
  output.
- When there are two numbers in the first column (for example 3997/1), it means
  that the function recursed 3997 times while it was called once.
- (x/y) for the first column says that y is the primitive calls and x is the
  recursive calls.
  
The column headings are interpreted as:
- ncalls: for the number of calls.
- tottime: for the total time spent in the given function (and excluding time
  made in calls to sub-functions)
- percall: is the quotient of tottime divided by ncalls
- cumtime: is the cumulative time spent in this and all subfunctions (from
  invocation till exit). This figure is accurate even for recursive functions.
- percall: is the quotient of cumtime divided by primitive calls
- filename: lineno(function) provides the respective data of each function

+++

#### Save output for `cProfile.run()`
```{note}
You can also save the run output to a file by providing it as the second
argument as follows:
```

```{code-cell} ipython3
cProfile.run('fib_dm(2000)', 'restats.log')
```

#### Command Line for cProfile

The files cProfile and profile can also be invoked using command line to profile
another python script containing multiple functions. But keep in mind, that 
profiling an entire script will include overhead time for importing, creating a function.

For Example, the code commented below can also create `restats.log` by profiling
`myscript.py` if there exists such a python script file in the directory.

```{code-cell} ipython3
## run using bash and saved `py` file
## !python -m cProfile -o restats.log myscript.py # myscript.py does not exist
```

#### `pstats.Stats` for output log of `cProfile()`

The pstats.Stats class reads profile results from a file and formats them in
readable manner.

```{code-cell} ipython3
# !pip install pstats   # pip install from within notebook using bash magic command
import pstats  ## Clean reprsentation of profile
from pstats import SortKey
p = pstats.Stats('restats.log') ## uses the above file created with output
p.print_stats()
```

```{code-cell} ipython3
### some possible print outputs
p.strip_dirs().sort_stats(-1).print_stats(.1)
p.sort_stats(SortKey.CUMULATIVE).print_stats(10) ## sort cumulative time spent
p.sort_stats(SortKey.TIME).print_stats(10)  ## sort time spent within each function
```

#### Deterministic Profiling of cProfile
Deterministic profiling is meant to reflect the fact that all function call,
function return, and exception events are monitored, and precise timings are
made for the intervals between these events (during which time the user’s code
is executing). In contrast, statistical profiling (which is not done by cprofile
module) randomly samples the effective instruction pointer, and deduces where
time is being spent. The latter technique traditionally involves less overhead
(as the code does not need to be instrumented), but provides only relative
indications of where time is being spent.


#### Interpretation of cProfile
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

```{code-cell} ipython3
# !pip3 install line_profiler  ## install line_profiler ## check if it is installed first
```

```{code-cell} ipython3
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

```{code-cell} ipython3
## plot stats profiling by percentage
## NEED to install gprof2dot (brew on mac)
!gprof2dot -f pstats restats.log | dot -Tsvg -o mine.svg
```

```{code-cell} ipython3
from IPython.display import SVG
SVG('mine.svg')

# # !pip install ipyplot
# import ipyplot

# ipyplot.plot_images( ['mine.svg'], # images should be passed in as an array
#     img_width=250,
#     force_b64=True # this is important to be able to render the image correctly on GitHub
# )
```

```{code-cell} ipython3
## remove files
!rm mine.svg
!rm restats.log
```

## Profiling Memory

- memory_profiler (open source python package)
- guppy

### Memory_Profiler

```{code-cell} ipython3
# !pip3 install -U memory_profiler    ## install memory_profiler
%load_ext memory_profiler
```

```{code-cell} ipython3
%%writefile memscript.py
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

```{code-cell} ipython3
from memscript import my_func
%mprun -T mprof0 -f my_func my_func()

## This goes into the pdb debugger as soon as 100MB is used
## !python3.9 -m memory_profiler --pdb-mmem=100 memscript.py  
```

```{code-cell} ipython3
print(open('mprof0', 'r').read())
```

```{code-cell} ipython3
## This can be improvised by mprof
# !pip3 install matplotlib
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
!mprof run memscript.py
!mprof plot -o image.svg

SVG('image.svg')
```

```{code-cell} ipython3
## Remove if not required to show.
!rm mprof0
!rm memscript.py
!rm image.svg
!rm mprofile_*
```

More information about this and even plots etc can be found at
[Memory_Profiler](https://pypi.org/project/memory-profiler/). 

In case you are running mprof as a command line, then using the @profile decorator
around a function or method and running python -m memory_profiler myscript will 
result in required work. You'll see line-by-line memory usage once your script exits.
