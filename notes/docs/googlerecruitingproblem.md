# Google recruiting problem
Find the first 10-digit prime found in consecutive digits of $e$ 

- Get the list of digits of $e$
- Write functions to achieve our goals(can be a loop, running over the list)
- Try to make it more efficient!


## Export $e$

If you try to export $e$ using the format command in Python, you will not get the precise $e$ value due to the double type data's feature!


```python
import math
print("%0.10f" % math.e)
print("%0.20f" % math.e)
```

    2.7182818285
    2.71828182845904509080
    

The digits after 16 positions of $e$ are wrong, compare this to the $e$ value export by "decimal".


```python
import operator
import decimal

#set the required digits
decimal.getcontext().prec = 150
e_decimal = decimal.Decimal(1).exp().to_eng_string()[2:]
print(e_decimal)
```

    71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526
    

Besides for the "decimal" module, we have lazy ways to find $e$ from an existed list. For example, the website https://apod.nasa.gov/htmltest/gifcity/e.2mil provides a list of $e$.


```python
import requests
# get text from the website
reply = requests.get('https://apod.nasa.gov/htmltest/gifcity/e.2mil').text

# remove the space in the lines
line_strip=[line.strip() for line in reply.split('\n')]

# connect all the digital lines
e=''.join([LINE for LINE in line_strip if LINE and LINE[0].isdigit()])
print(e[:20:])
```

    2.718281828459045235
    

## Write functions to check whether a number is a prime
Here the most basic and bruteforce way is used. Check all the factors less than $\sqrt n+1$ for all the odd numbers $n$.\
We have many powerful algorithms to achieve this if we need to check for very large numbers. For this problem, this method is efficcient enough.


```python
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

    7427466391
    Wall time: 8.98 ms
    
