from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import re
# html=urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
# bsObj=BeautifulSoup(html.read(),'lxml')
# print(bsObj.h1)
# nameList=bsObj.findAll("span",{"class":"green"})
# for name in nameList:
#     print(name.get_text())

# html=urlopen("http://en.wikipedia.org/wiki/Kevin_Bacon")
# bsObj=BeautifulSoup(html,'lxml')
# for link in bsObj.findAll("a"):
#     if 'href' in link.attrs:
#         print(link.attrs['href'])


# from requests_html import HTMLSession
# session=HTMLSession()
# url='https://www.jianshu.com/p/85f4624485b9'
# r=session.get(url)
# # print(r.html.text)
# # print(r.html.links)
# print(r.html.absolute_links)
# sel = 'body > div.note > div.post > div.article > div.show-content > div > p:nth-child(6) > a'
# # print(get_text_link_from_sel(sel))

a=[1,2]
b=[3,4]
c=a+b
print(c)