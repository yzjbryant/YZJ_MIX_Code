cache = {}

def get_page(url):
    if cache.get(url):
        return cache[url]  #返回缓存的数据
    else:
        data = get_data_from_server(url)
        cache[url] = data      #先将数据保存到缓存中
        return data