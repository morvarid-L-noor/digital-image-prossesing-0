# -*- coding: utf-8 -*-
"""Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1svCd4OsW_jhIOxbEicLZ_Xu3yiV1QHb4
"""

#Q2-------------------------------------------------------------------------------
import random

def random_generate():
  dictionary = {}
  res = [] 

  for j in range(100): 
      res.append(random.randint(-2, 7))   
  return res

def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count

random_list = random_generate()
tmp = {}
dct ={}
w = 1
print(random_list)
max_list = max(random_list)
for m in range(10):
  for i in random_list: 
    if(i == max_list):
      count = countX(random_list,i)
      tmp[max_list] = count
      if(count != 0):
        for n in range(count):
          random_list.remove(i)
        if (len(random_list) != 0):
          max_list = max(random_list)
        else:
          break 
    
print(tmp)
for x, y in tmp.items() :
  lst = []
  for i in range(y):
    lst.append(x)
  dct[w] = lst
  w += y

print(dct)