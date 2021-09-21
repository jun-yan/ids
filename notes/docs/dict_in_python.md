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

    a
    2
    [1, 2, 3]
    


```python
tuple1=(1,2,3)
print(tuple1[2])
tuple1[2]=100
```

    3
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_5908/1596880032.py in <module>
          1 tuple1=(1,2,3)
          2 print(tuple1[2])
    ----> 3 tuple1[2]=100
    

    TypeError: 'tuple' object does not support item assignment



```python
a=range(5)
print(a)
b=[i for i in range(5)]
print(b)
```

    range(0, 5)
    [0, 1, 2, 3, 4]
    

## Mapping type: dictionary
A set of value pairs\
Indexed by "keys": a immutable type, should be unique\
Tuple can be used as keys\



```python
score={'tom':90,'jack':80}
score['alice']=70
score
```




    {'tom': 90, 'jack': 80, 'alice': 70}




```python
del socre['tom']
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_5908/3109481407.py in <module>
    ----> 1 del socre['tom']
    

    NameError: name 'socre' is not defined



```python
score['jack']=100
score
```




    {'tom': 90, 'jack': 100, 'alice': 70}



Construct dictionary from list of pairs:


```python
dict(tom=80,jack=90)
```




    {'tom': 80, 'jack': 90}



## Some useful functions:


```python
score.items()
```




    dict_items([('tom', 90), ('jack', 100), ('alice', 70)])




```python
score.keys()
```




    dict_keys(['tom', 'jack', 'alice'])




```python
score.values()
```




    dict_values([90, 100, 70])




```python
# check for the key
print(score.get('steven','not found'))
print(score.get('alice','not found'))
```

    not found
    70
    

## Applications:
Find the winner of an election:
The results is {"john", "johnny", "jackie", "johnny", "john", "jackie", "jamie", "jamie", "john", "johnny", "jamie", "johnny", "john"}\
Try to write a function to out put people and their votes.


```python
dict1=dict(jimmy=3,john=1)
data = ["john", "johnny", "jackie", "johnny", "john", "jackie", "jamie", "jamie", "john","johnny", "jamie", "johnny",  "john"]
def vote(data):
    Mydict=dict()
    for i in data:
        if i not in Mydict.keys():
            Mydict[i]=0
        if i in Mydict.keys():
            Mydict[i]+=1
            
    return Mydict 
print(vote(data))
```

    {'john': 4, 'johnny': 4, 'jackie': 2, 'jamie': 3}
    
