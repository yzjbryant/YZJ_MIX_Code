#请求网页
import requests
import re
import os
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
response=requests.get('https://www.vmgirls.com/12985.html',headers=headers)
# print(response.request.headers)
html=response.text
# print(html)

#解析网页
dir_name=re.findall('<h1 class="post-title h3">(.*?)</h1>',html)[-1]
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
urls=re.findall('<a href="(.*?)" alt=".*?" title=".*?">',html)
print(urls)

for url in urls:
    #图片的名字
    import time
    time.sleep(1)
    file_name=url.split('/')[-1]
    response=requests.get(url,headers=headers)
    with open(dir_name+'/'+file_name,'wb') as f:
        f.write(response.content)
