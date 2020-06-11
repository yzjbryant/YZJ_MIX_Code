def fact(x):
    if x == 1:
        return
    else:
        return x * fact(x-1)

def sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

print sum([1,2,3,4])