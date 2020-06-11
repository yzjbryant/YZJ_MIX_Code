# 定义一个函数sum_numbers
# 能够接受一个num的整数参数
# 计算1+2+...num的结果

def sum_numbers(num):

    #1.出口
    if num == 1:
        return 1

    #2.数字的累加 num + (1...num-1)
    #假设sun_numbers 能够正确的处理1...num - 1
    temp = sum_numbers(num - 1)


    #两个数字的相加
    return num + temp

result = sum_numbers(1000)
print(result)

