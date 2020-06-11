from collections import deque
search_queue = deque()        #创建一个队列
search_queue += graph["you"]  #将你的邻居都加入到这个搜索队列中

while search_queue:   #只要队列不为空
    person = search_queue.popleft() #就取出其中的第一个人
    if person_is_seller(person): #检查这个人是否是芒果销售商
        print(person + "is a mango seller!")  #是芒果销售商
        return True
    else:
        search_queue += graph[person]  #不是芒果销售商，将这个人的朋友都加入搜索队列

return False  #如果到达了这里，就说明队列中没人是芒果销售商

def person_is_seller(name):
    return name[-1] == 'm'