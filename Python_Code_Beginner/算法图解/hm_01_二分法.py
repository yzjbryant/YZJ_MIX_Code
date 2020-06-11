def binary_search(my_list,item):
    low = 0   # (以下2行)low和high用于跟踪要在其中查找的列表部分
    high = len(my_list) - 1

    while low <= high:   #只要范围没有缩小到只包含到一个元素，
        mid = (low + high)/2  #就检查中间的元素
        guess = my_list[mid]
        if guess == item:   #找到了元素
            return mid
        if guess > item:    #猜的数字大了
            high = mid - 1
        else:               #猜的数字小了
            low = mid + 1
    return None     #没有指定的元素

my_list = [1,3,5,7,9]       #测试

S1 = binary_search(my_list,3)
S2 = binary_search(my_list,-1)
print("查找到的S1 = %f" %S1) # =>1  别忘了索引从0开始，第二个位置的索引为1
print("查找到的S2 = %f" %S2) # >= None 在Python中，None表示空，它意味着没有找到指定的元素
