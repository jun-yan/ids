---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Python dictionary



## Python sequence types:
To store collection of data. They are sequential so we can iterate over them. Some can be accessed by an integar index.
- List:
mutable, use a square brackets $[ \quad]$
- Tuple:
immutable, use a parentheses $(\quad)$
- Range:
sequence of integars

```python
list1=[1,"a",[1,2,3]]
print(list1[1])
list1[0]=2
print(list1[0])
print(list1[-1])
```

```python
tuple1=(1,2,3)
print(tuple1[2])
try:
    tuple1[2]=100
except:
    print("error")
```

```python
a=range(5)
print(a)
b=[i for i in range(5)]
print(b)
```

## Mapping type: dictionary
A set of value pairs\
Indexed by "keys": a immutable type, should be unique\
Tuple can be used as keys


```python
score={'tom':90,'jack':80}
score['alice']=70
score
```

```python
del score['tom']
```

```python
score['jack']=100
score
```

Construct dictionary from list of pairs:

```python
dict(tom=80,jack=90)
```

Construct dictionart using zip()

```python
list1=["Tom","Jack","Steve"]
list2=[5,7,9]
zip_pairs=zip(list1,list2)
print(zip_pairs)
dict_pairs=dict(zip_pairs)
print(dict_pairs)
```

## Some useful functions:

```python
score.items()
```

```python
score.keys()
```

```python
score.values()
```

```python
# check for the key
print(score.get('steven','not found'))
print(score.get('alice','not found'))
```

```python
"jack" in score
```

```python
"tom" not in score
```

## Applications:
Find the winner of an election:
The results is {"john", "johnny", "jackie", "johnny", "john", "jackie", "jamie", "jamie", "john", "johnny", "jamie", "johnny", "john"}\
Try to write a function to out put people and their votes.

```python
dict1=dict(jimmy=3,john=1)
data = ["john", "johnny", "jackie", "johnny", "john", "jackie", "jamie", "jamie", "john","johnny", "jamie", "johnny",  "john"]
def vote(data,Mydict):
    for i in data:
        if i not in Mydict.keys():
            Mydict[i]=0
        if i in Mydict.keys():
            Mydict[i]+=1           
    return Mydict 

vote(data,dict1)
```

## Merging dictionaries

```python
fruit = {"A": {"Apple"}, "B":{"Grape","Pear"}}
fruit2 = {"A":{"Banana"}, "C":{"Pear"}}
fruit.update(fruit2)
fruit
```

## Iterating over dictionaries

```python
%%timeit  d = {"a":123, "b":34, "c":304, "d":99}
for key in d.keys():
    x = d[key]

```

```python
%%timeit  d = {"a":123, "b":34, "c":304, "d":99}
for value in d.values():
    x = value
```
