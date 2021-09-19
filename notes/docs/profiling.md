# Profiling and Timing 

### Hidden Code: (Nothing to do with Profiling)


```python
#@hidden
from IPython.display import HTML, Image

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('.cm-comment:contains(@hidden)').closest('div.input').hide();
 } else {
 $('.cm-comment:contains(@hidden)').closest('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for the following figures related code is by default hidden. This can be used to give
quizzes whereby the raw dataset is hidden but a plot is shown or so on.
To toggle on/off the raw code, <a href="javascript:code_toggle()">click here to toggle for raw hidden code</a>.''')
```




### Motivation for Profiling
Speed? Jump the bottlenecks
Which one do you want your code to be? 


```python
#@hidden

## The above comment hides this cell

def _src_from_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'

def gallery(images, captions, row_height, row_width):
    """Shows a set of images in a gallery that flexes with the width of the notebook.
    
    Parameters
    ----------
    images: list of str or bytes
        URLs or bytes of images to display
    
    captions: list of captions

    row_height: str
        CSS height value to assign to all images. Set to 'auto' by default to show images
        with their native dimensions. Set to a value like '250px' to make all rows
        in the gallery equal height.
    """
    figures = []
    for i in range(len(images)):
        image = images[i]
        if isinstance(image, bytes):
            src = _src_from_data(image)
            caption = captions[i]
        else:
            src = image
            caption = captions[i]
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" height="{row_height}" width="{row_width}">
              {caption}
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

gallery( ["https://www.seekpng.com/png/detail/127-1274961_visit-zootopia-jpg-free-sloth-characters-in-zootopia.png",
        "https://freepngimg.com/thumb/the_flash/5-2-flash-free-png-image.png"], ["Slow", "Fast"], 400,  400)
```





This could be one most common reason for timing and profiling. The understanding is that only some small chunk often makes our code heavy in terms of time taken, memory usage, etc. Finding it and improving or managing it can improve our resource usage.

It is especially important to profile the code chunks to optimize. This becomes more important in a production environment where resources are expensive and cannot be wasted. Therefore, as part of testing, one should include profiling to see which chunks use most resources and which chunks use less. Some coding contests often have a maximum time to run or people boast of how fast their particular solution runs

### Resources mostly profiled
The most common resources that we often optimize for projects are: 
- CPU time
- Memory Usage

But some other resources could also be network bandwidth usage etc. For example, if you wish to create a notebook that gathers stock data from one of many locations such as Yahoo Finance etc, and you only have a limited bandwidth, your choice could also depend on where you gather your data from.

### Profiling CPU time
Profiling implies that it divides the entire code file into smaller code chunks (*functions/lines*) in order to show the  resource usage and thus existence of bottlenecks in our code. It is best practice to find bottlenecks by means of profiling rather than by intuition. 

Python includes the following modules that achieve profiling for CPU time:
- timeit
- cprofile
- profile
- lineprofilee

For interpretation, we use the fibonacci example shown in class:

**Note: The profiler modules are designed to provide an execution profile for a given program, not for benchmarking purposes (for that, there is timeit for reasonably accurate results). This particularly applies to benchmarking Python code against C code: the profilers introduce overhead for Python code, but not for C-level functions, and so the C code would seem faster than any Python one.**

### Timeit Function Usage:


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

## One  liners can be done using timeit magic function %timeit (this is a feature of ipython)
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

This is a great tool (in ipython) as you can work on some code functions and applications and then save each chunk correspondingly to the "source" file with functions and "application" file once you ensure it works on the notebook. It helps a great deal as you'd like your functions to finally appear in a ".py" file to automate it in clusters or so on. 


```python
!cp myscript.py myscript2.py
%save -a myscript2.py -n 4   ##(-a for append, -n for the line run (changes every session,
                            ## good for mostly dynamic usage))
```

### Command Line Interface for timeit


```python
!python -m timeit -r 20 '"-".join(str(n) for n in range(100))'  ## r does repetition
## The output suggests there were 20000 loops, repeated 20 times for accuracy, and 

"-".join(str(n) for n in range(100))
```

## cProfile for profiling CPU time

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

- The first line indicates that 21 calls were monitored of which 5 were primitive calls (calls that are not induced via recursion). 
- "Ordered by: standard name" suggests usage of the last column for sorting the output.
- When there are two numbers in the first column (for example 3/1), it means that the function recursed.
- (x/y) for the first column says that y is the primitive calls and x is the recursive calls.
- You can also save the run output to a file by providing it as the second argument as follows:


```python
cProfile.run('fib_dm(2000)', 'restats')
```

### Command Line for cProfile

The files cProfile and profile can also be invoked as a script to profile another script. For example:


```python
### Run it afterwards
!python -m cProfile -o restats myscript.py   ## keep in mind that if i profile an entire script, it will include 
                                             ## the overhead time for importing, creating a function and so on.
## Once again run at the end
cProfile.run('fib_dm(2000)', 'restats')
```

## Interpretation of Raw Output format:

The column headings include:
- ncalls: for the number of calls.
- tottime: for the total time spent in the given function (and excluding time made in calls to sub-functions)
- percall: is the quotient of tottime divided by ncalls
- cumtime: is the cumulative time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.
- percall: is the quotient of cumtime divided by primitive calls
- filename: lineno(function) provides the respective data of each function

The pstats.Stats class reads profile results from a file and formats them in readable manner. 


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
Deterministic profiling is meant to reflect the fact that all function call, function return, and exception events are monitored, and precise timings are made for the intervals between these events (during which time the user’s code is executing). In contrast, statistical profiling (which is not done by cprofile module) randomly samples the effective instruction pointer, and deduces where time is being spent. The latter technique traditionally involves less overhead (as the code does not need to be instrumented), but provides only relative indications of where time is being spent.


### Interpretation of cProfile
Call count statistics can be used to identify bugs in code (surprising counts), and to identify possible inline-expansion points (high call counts). Internal time statistics can be used to identify “hot loops” that should be carefully optimized. Cumulative time statistics should be used to identify high level errors in the selection of algorithms. Note that the unusual handling of cumulative times in this profiler allows statistics for recursive implementations of algorithms to be directly compared to iterative implementations.

### Line Profiler

*line_profiler will profile the time individual lines of code take to execute. The profiler is implemented in C via Cython in order to reduce the overhead of profiling.* Also the timer unit is $10^{-6}$ or $\mu s$


```python
!pip3 install line_profiler  ## install line_profiler
```


```python
from line_profiler import LineProfiler
import random

def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]    ## you can also use for loop in a single line after operation
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
!python3.9 -m memory_profiler example.py   ## using to find the memory used by my function 
## !python3.9 -m memory_profiler --pdb-mmem=100 example.py  ## This goes into the pdb debugger as soon as 100MB is used
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

More information about this and even plots etc can be found at [Memory_Profiler](https://pypi.org/project/memory-profiler/). You can use a memory profiling by putting the @profile decorator around any function or method and running python -m memory_profiler myscript. You'll see line-by-line memory usage once your script exits.

### Guppy

There are number of ways to profile an entire Python application. Most python way is Guppy. You can take a snapshot of the heap before and after a critical process. Then compare the total memory and pinpoint possible memory spikes involved within common objects. Well, but apparently it works with Python 2.


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
objgraph.show_most_common_types()   ## gives a quick overview of the objects in memory
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

It’s easy to see MyBigFatObject instances that appeared and were not freed. So, we can  trace the reference chain back.
