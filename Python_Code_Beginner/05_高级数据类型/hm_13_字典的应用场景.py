# 使用多个键值对，存储描述一个物体的相关信息-描述更复杂的数据信息
# 将 多个字典 放在 一个列表 中，在进行遍历
card_list = [
    {"name":"zhangsan",
     "qq":"1223456",
    "phone":"110"},
    {"name":"李四",
     "qq":"122346",
    "phone":"112"}
]

for card_info in card_list:

    print(card_info)
