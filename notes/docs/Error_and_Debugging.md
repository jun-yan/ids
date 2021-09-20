# Error and Debugging
Resource: Python Data Science Handbook

Code development and data analysis always require a bit of trial and error, and IPython contains tools to streamline this process. 

# Error


```python
def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x + 1
    return func1(a, b)
```


```python
func2(-1)
# import pdb;
# pdb.set_trace()
# ;
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_22380/332324174.py in <module>
    ----> 1 func2(-1)
          2 # import pdb;
          3 # pdb.set_trace()
          4 # ;
    

    ~\AppData\Local\Temp/ipykernel_22380/503927240.py in func2(x)
          5     a = x
          6     b = x + 1
    ----> 7     return func1(a, b)
    

    ~\AppData\Local\Temp/ipykernel_22380/503927240.py in func1(a, b)
          1 def func1(a, b):
    ----> 2     return a / b
          3 
          4 def func2(x):
          5     a = x
    

    ZeroDivisionError: division by zero


Calling func2 results in an error, and reading the printed trace lets us see exactly what happened. By default, this trace includes several lines showing the context of each step that led to the error.

# Debugging

The standard Python tool for interactive debugging is pdb, the Python debugger. This debugger lets the user step through the code line by line in order to see what might be causing a more difficult error. The IPython-enhanced version of this is ipdb, the IPython debugger.


```python
%debug
```

    > [1;32mc:\users\jiz52\appdata\local\temp\ipykernel_22380\503927240.py[0m(2)[0;36mfunc1[1;34m()[0m
    
    ipdb> print(a)
    -1
    ipdb> print(b)
    0
    ipdb> q
    

The interactive debugger allows much more than this, though–we can even step up and down through the stack and explore the values of variables there

# Partial list of debugging commands
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
