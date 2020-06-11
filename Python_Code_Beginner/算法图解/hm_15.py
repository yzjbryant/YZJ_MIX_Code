def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []   #这个数组用于记录检查过的人
    while search_queue:
        person = search_queue.popleft()
        if person not in searched:  #仅当这个人没检查过时才检查
            if person_is_seller(person):
                print(person + "is a mango seller")
                return True
            else:
                search_queue += graph[person]
                search.append(person)  #将这个人标记为检查过
    return False

search("you")
