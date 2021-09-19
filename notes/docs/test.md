---
jupytext:
  formats: ipynb,md:myst,py:light
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

# My nifty title
Some **text**!


This is similar to any type of Markdown file, thus typical R Markdown 
structure which has a mixture of Latex. 

For example to create headers:

### Large Header
## Larger Header
# Largest Header


MyST - Markedly Structured Text is a Python. It allows the same 
functionality as above but also creating directives.
Essentially creating a statement that will be eye-catching for the viewer.

There are 4 components when writing a directive:
Name of directive -  similar to a function need to define what 
you would like create. This would need to be written within {} brackets.

For example, say we want to create a link for GitHub repository for class, 
adding [Introduction Data Science](https://github.com/jun-yan/ids)


```{admonition} Add my admonition
Adding my little admonition
```

````{note}
Initial
```{warning}
warning
```
````

```{eval-rst}
.. note::

   A note written in reStructuredText.
```

```{code-cell} ipython3
---
other:
  more: true
tags: [hide-output, show-input]
---
print("Hello!")
```

```{code-block} python
---
lineno-start: 10
emphasize-lines: 1, 3
caption: |
    This is my
    multi-line caption. It is *pretty nifty* 
---
a = 2
print('my 1st line')
print(f'my {a}nd line')
```

:::{note}
**Wow**, a note!
:::

```{admonition} Here's my title
:class: warning

Here's my admonition content
```

```{math} ax^{2} + bx + c
---
label: quadratic
---
```

The basic quadratic equation, {math:numref}`quadratic`, allows for the 
construction of all kinds of parabolas
