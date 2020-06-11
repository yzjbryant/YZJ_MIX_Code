sum [] = 0                  #基线条件
sum (x:xs) = x + (sum xs)   #递归条件

sum arr = if arr == []
    then 0
    else (head arr) + (sum(tail arr))

def Calculatelist(x,xs):
    sum [] = 0
    sum(x,xs) = x + (sum xs)

list = [1,2,3,4]

sum = 0

i = 0
while i <=100:
    print(i)
    sum  += i
    i+=1
print("sjkfhsjk%d"%sum)

list = [1,2,3,4]



