import requests
import re
import lxml
import os
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}

#爬取全国34个省网址
response=requests.get('http://www.gx211.com/gxmd/gx-bj.html',headers=headers)
html=response.text
url=re.findall('href="(.*?)"',html)
Name_of_Province=['北京','天津','河北','山西','辽宁','吉林','黑龙江','上海','江苏','浙江','安徽','福建','江西',
                '山东','河南','湖北','湖南','广东','内蒙古','广西','海南','重庆','四川','贵州','云南','西藏',
                '陕西','甘肃','青海','宁夏','新疆','港澳台']
Province_34_website=[]
for i in url:
    if 'gx-' in i:
        Province_34_website.append('http://www.gx211.com/gxmd/%s'%i)
Province_34_website.append('http://www.gx211.com/gxmd/gx-gat.html')

#获取各省的所有高校
'''
# for i in range(len(Province_34_website)):
for i in Province_34_website:
    response_province=requests.get(i,headers=headers)
    response_province.encoding="gb2312"
    html=response_province.text
    url=re.findall('<td style="text-align: left; text-indent: 20px"><a target="_blank" href="(.*?)/"',html)
    print(len(url))
'''

#通用函数
def each_province(url):
    response=requests.get(url,headers=headers)
    response.encoding="gb2312"
    html=response.text
    return html

#北京市——未成功
url_beijing=re.findall('<td style="text-align: left; text-indent: 20px"><a target="_blank" href="(.*?)">',each_province(Province_34_website[0]))
# print(len(url_beijing),url_beijing)

#天津市——
# url_tianjin=re.findall('<td style="text-align: left; text-indent: 20px"><a target="_blank" href="(.*?)">',each_province(Province_34_website[1]))
url_tianjin=re.findall('<a href="(.*?)" target="_blank">',each_province(Province_34_website[1]))
print(len(url_tianjin),url_tianjin)




#港澳台——成功
response_gat=requests.get(Province_34_website[-1],headers=headers)
response_gat.encoding="gb2312"
html=response_gat.text
url=re.findall('(.*?)<a href="(.*?)" target="_blank">(.*?)',html)
# print("港澳台：",url)
























#江苏省
'''
response=requests.get('http://www.gx211.com/gxmd/gx-js.html',headers=headers)
# print(response.request.headers)
html=response.text
# print(html)
#解析网页
# dir_name=re.findall('<h1 class="post-title h3">(.*?)</h1>',html)[-1]
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
urls1=re.findall('/">(.*?)</a></td>',html)
urls2=re.findall('target="_blank">(.*?)</a>',html)
del urls2[-1]
del urls2[-1]
urls3=re.findall('<a href="http://szlg.just.edu.cn/">(.*?)</a></td>',html)
urls_all=urls1+urls2+urls3
print(len(urls_all),urls_all)
'''


