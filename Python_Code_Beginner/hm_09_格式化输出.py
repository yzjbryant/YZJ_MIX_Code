# 定义字符串变量name,输出 我的名字叫 小明，请多多关照！
name = "大!明"
print("我的名字叫 %s，请多多关照！"% name)

student_no = 100123
print("我的学号是%07d"%student_no)

price = 8.5
weight = 7.5
money = price * weight
print("苹果单价：%0.3f元/斤，购买了：%0.4f斤，需要支付：%0.1f元"%(price,weight,money))

scale = 0.25
print("数据比例是%.2f%%"%(scale * 100))