import requests
from bs4 import BeautifulSoup
import re
from urllib.request import urlopen,Request
import urllib.request

#  Sample 1
# url='http://www.cntour.cn/'
# strhtml=requests.get(url)
# soup=BeautifulSoup(strhtml.text,'lxml')
# # print(soup)
# print("++++++++++++++++")
# data=soup.select('#main>div>div.mtop.firstMod.clearfix>div.centerBox>ul.newsList>li>a')
# # print(data)
# for item in data:
#     result={'title':item.get_text(),
#             'link':item.get('href'),
#             'ID':re.findall('\d+',item.get('href'))}
#     print(result)

# Sample 2
#抓取超链接
# url='https://www.douban.com/'
# headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
# ret=Request(url,headers=headers)
# html=urlopen(ret)
# bsObj=BeautifulSoup(html,'html.parser')
# #抓取超链接
# t1=bsObj.find_all('a') #  a为超链接
# for t2 in t1:
#     t3=t2.get('href')
#     print(t3)
# #抓取图片链接
# t1=bsObj.find_all('img')
# for t2 in t1:
#     t3=t2.get('src')
#     print(t3)

# Sample 3
#把一个网页中所有链接地址提取出来（去重）
url="http://blog.csdn.net"
pattern1='<.*?(href=".*?").*?'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
opener=urllib.request.build_opener()
opener.addheaders=[headers]
data=opener.open(url).read().decode('utf8')
#提取链接
content_href=re.findall(pattern1,data,re.I)
print(content_href)
#过滤重复链接
set1=set(content_href)
#写文件
file_new="./href.txt"
with open(file_new,'w') as f:
    for i in set1:
        f.write(i)
        f.write("\n")
