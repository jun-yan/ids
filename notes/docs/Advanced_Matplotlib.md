---
jupytext:
  formats: ipynb,md:myst
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

## Advanced Matplotlib

+++

### Customizing Ticks

+++

Matplotlib’s default tick locators and formatters are designed to be generally sufficient in many situations, 
but are not suitable for every plot.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

#### Hiding Ticks or Lables

```{code-cell} ipython3
ax = plt.axes()
ax.plot(np.random.rand(100))
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())
```

```{code-cell} ipython3
fig, ax = plt.subplots(4, 4, figsize=(4, 4))
fig.subplots_adjust(hspace=0, wspace=0)
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images
for i in range(4):
    for j in range(4):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap="bone")
```

#### Reducing or Increasing the Number of Ticks

```{code-cell} ipython3
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
```

```{code-cell} ipython3
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(5))
    axi.yaxis.set_major_locator(plt.MaxNLocator(5))
fig
```

#### Fancy Tick Formats

```{code-cell} ipython3
# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 2000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')
# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi)
```

```{code-cell} ipython3
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig
```

```{code-cell} ipython3
def format_func(value, tick_number):
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig
```

### Customizing Matplotlib: Configurations and Stylesheets

+++

#### Stylesheets

```{code-cell} ipython3
plt.style.available[:5]
```

```{code-cell} ipython3
plt.style.use('classic')
def hist_and_lines():
    np.random.seed(0)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(np.random.randn(1000))
for i in range(3):
    ax[1].plot(np.random.rand(20))
    ax[1].legend(['x', 'y', 'z'], loc='lower left')
```

### Three-Dimensional Plotting in Matplotlib

```{code-cell} ipython3
from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
```

#### Three-Dimensional Points and Lines

```{code-cell} ipython3
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, 25, 2000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
# Data for three-dimensional scattered points
zdata = 15 * np.random.random(200)
xdata = np.sin(zdata) + 0.1 * np.random.randn(200)
ydata = np.cos(zdata) + 0.1 * np.random.randn(200)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
```

#### Three-Dimensional Contour Plots

```{code-cell} ipython3
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6, 6, 50)
y = np.linspace(-6, 6, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
```

```{code-cell} ipython3
ax.view_init(50, 10)
fig
```

#### Wireframes Plots

```{code-cell} ipython3
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')
```

#### Surface Plots

```{code-cell} ipython3
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
cmap='viridis', edgecolor='none')
ax.set_title('surface')
```

#### Surface Triangulations

```{code-cell} ipython3
theta = 2 * np.pi * np.random.random(3000)
r = 6 * np.random.random(3000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
```

```{code-cell} ipython3
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
cmap='viridis', edgecolor='none')
```

### Geographic Data with Basemap

+++

Use this in order to download the package

```{code-cell} ipython3
#conda install -c conda-forge basemap-data-hires=1.0.8.dev0
#or
#conda install basemap
```

```{code-cell} ipython3
import os
os.environ['PROJ_LIB'] = r'C:\Users\astro\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
```

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
```

```{code-cell} ipython3
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,width=8E6, height=8E6,lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)
# Map (long, lat) to (x, y) for plotting
x, y = m(-72.3, 41.8)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, 'UConn', fontsize=12)
```

#### Map Projections

```{code-cell} ipython3
from itertools import chain
def draw_map(m, scale=0.2):
    m.shadedrelief(scale=scale)
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
```

```{code-cell} ipython3
## Cylindrical projections

fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None,llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)
```

More types of projections can be found here: https://matplotlib.org/basemap/users/mapsetup.html

+++

#### Drawing a Map Background

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
for i, res in enumerate(['l', 'h']):
    m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2,
    width=90000, height=120000, resolution=res, ax=ax[i])
    m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
    m.drawmapboundary(fill_color="#DDEEFF")
    m.drawcoastlines()
    ax[i].set_title("resolution='{0}'".format(res))
```

Left side is low resolution and right side is high resolution images.

+++

References:
    
[1] Jake VanderPlas, Python Data Science Handbook

[2] Matplotlib Basemap Toolkit documentation, https://matplotlib.org/basemap/
