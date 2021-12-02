---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Matplotlib


reference: [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do)


**Matplotlib** is a multi-platform data visualization library built on NumPy arrays, and designed to work with the broader SciPy stack.  
It was conceived by John Hunter in 2002,  
originally as a patch to IPython for enabling interactive MATLAB-style plotting via gnuplot from the IPython command line.   


One of Matplotlib’s most important features is its ability to play well with many operating systems and graphics backends. 


## Importing Matplotlib

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
```

## Setting Styles


Here we will set the classic style, which ensures that the plots we create use the classic Matplotlib style:

```{code-cell} ipython3
plt.style.use('classic')
```

### Plotting from a script

```{code-cell} ipython3
# ------- file: myplot.py ------
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()

```

You can then run this script from the command-line prompt, which will result in a window opening with your figure displayed:  

$ python myplot.py


### Plotting from an IPython notebook

<!-- #region -->
Plotting interactively within an IPython notebook can be done with the %matplotlib command,  
and works in a similar way to the IPython shell. In the IPython notebook,   
you also have the option of embedding graphics directly in the notebook, with two possible options:  


**%matplotlib** notebook will lead to interactive plots embedded within the notebook  

**%matplotlib inline** will lead to static images of your plot embedded in the notebook  
For this book, we will generally opt for **%matplotlib inline**:
<!-- #endregion -->

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
```

## Saving Figures to File

```{code-cell} ipython3
fig.savefig('my_figure.tif')
```

To confirm that it contains what we think it contains, let's use the IPython Image object to display the contents of this file:  


```{code-cell} ipython3
from IPython.display import SVG
SVG('my_figure.svg')
```

## Two Interfaces


**MATLAB-style Interface**  
Matplotlib was originally written as a Python alternative for MATLAB users, and much of its syntax reflects that fact.  
The MATLAB-style tools are contained in the pyplot (plt) interface.  
For example, the following code will probably look quite familiar to MATLAB users:  

```{code-cell} ipython3
plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));
```

**Object-oriented interface**  
The object-oriented interface is available for these more complicated situations,  
and for when you want more control over your figure.   
Rather than depending on some notion of an "active" figure or axes,  
in the object-oriented interface the plotting functions are methods of explicit Figure and Axes objects.  
To re-create the previous plot using this style of plotting, you might do the following:

```{code-cell} ipython3
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
```

## Line Plots

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

```{code-cell} ipython3
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));
fig.savefig('lineplot1.svg')
```

```{code-cell} ipython3
# change colors
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
```

```{code-cell} ipython3
# change linestyle
# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted
```

```{code-cell} ipython3
# change Axes Limits
plt.plot(x, np.sin(x))

plt.xlim(5, 8)
plt.ylim(-0.5, 0.5);
```

```{code-cell} ipython3
# alternative 
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);
```

```{code-cell} ipython3
# add Titles and axis labels
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)");
```

```{code-cell} ipython3
# add legend
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend();
```

## Scatter Plot

```{code-cell} ipython3
# Different Marker
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);
```

```{code-cell} ipython3
# Different size
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();  # show color scale
```

## Visualizing Errors


errorbar(x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, *, data=None, **kwargs)

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

```{code-cell} ipython3
x = np.linspace(0, 10, 50)
dy = 2*y
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);
```

### Continuous Error

```{code-cell} ipython3
from sklearn.gaussian_process import GaussianProcessRegressor

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcessRegressor()
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, dyfit_ori = gp.predict(xfit[:, np.newaxis],return_std=True)
dyfit = 2 * dyfit_ori  # 2*sigma ~ 95% confidence region
```

```{code-cell} ipython3
# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')

plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.xlim(0, 10);
```

## Contour Plots


matplotlib.pyplot.contour(*args, data=None, **kwargs)

```{code-cell} ipython3
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```

```{code-cell} ipython3
x = np.linspace(0, 5, 51)
y = np.linspace(0, 5, 41)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
```

```{code-cell} ipython3
plt.contour(X, Y, Z, colors='black');
```

```{code-cell} ipython3
plt.contourf(X, Y, Z, 20, cmap='jet')
plt.colorbar();
```

## Histograms, Binnings, and Density


hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, *, data=None, **kwargs)

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = np.random.randn(1000)
```

```{code-cell} ipython3
plt.hist(data, bins=30, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');
```

```{code-cell} ipython3

x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3,  bins=40)

plt.hist(x1, **kwargs);
plt.hist(x2, **kwargs);
plt.hist(x3, **kwargs);
```

```{code-cell} ipython3
# One way to plot figures separately
plt.figure()
x1 = np.random.normal(0, 0.8, 1000)
plt.hist(x1)
plt.figure()
x2 = np.random.normal(0, 3, 1000)
plt.hist(x2);
```

### Two-Dimensional Histograms and Binnings


hist2d(x, y, bins=10, range=None, density=False, weights=None, cmin=None, cmax=None, *, data=None, **kwargs)

```{code-cell} ipython3
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
```

```{code-cell} ipython3
plt.hist2d(x, y,bins=100,density=True, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
```

```{code-cell} ipython3
a=plt.hist(x,density=True);
plt.close()
plt.figure()
plt.plot(a[1][1:],a[0])     
```
