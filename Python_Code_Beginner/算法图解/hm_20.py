fruits = set(["avocado","tomato","banana"])
vegetables = set(["beets","carrots","tomato"])
fruits | vegetables   #并集
fruits & vegetables   #交集
fruits - vegetables   #差集
vegetables - fruits


print(fruits | vegetables)
print(fruits & vegetables)
print(fruits - vegetables)
print(vegetables - fruits)