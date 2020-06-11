# 定义布尔型变量 has_ticket 表示是否有车票
has_ticket = False

# 定义整数型变量 knife_length 表示刀的长度，单位：厘米
knife_length = 30

# 首先检查是否有车票，如果有，才允许进行 安检
if has_ticket:
    print("车票检查通过，准备开始安检")

    # 安检时，需要检查刀的长度，判断是否超过20厘米
    if knife_length > 20:

        # 如果超过20厘米，提示刀的长度，不允许上车
        print("你携带的刀太长，有 %d 公分长"% knife_length)
        print("不允许上车")

    # 如果不超过20厘米，安检通过
    else:
        print("安检已经通过，祝您旅途愉快")

# 如果没有车票，不允许进门
else:
    print("大哥请先买票")