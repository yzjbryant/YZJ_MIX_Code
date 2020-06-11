import pyquery
import re
import logging
import pymongo
import requests
from pyquery import PyQuery as pq
from urllib.parse import urljoin
'''
# r=requests.get('https://github.com/favicon.ico',verify=False,timeout=1)
# header=''
# with open('favicon.ico','wb') as f:
#     f.write(r.content)
# r2=requests.get('https://qq.com/',auth=HTTPBasicAuth('admin','admin'))
# print(r.status_code)
# cilent=pymongo.MongoClient(host='localhost',port=27017)
'''
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s')
BASE_URL='https://static1.scrape.cuiqingcai.com'
TOTAL_PAGE=10
#页面爬取
def scrape_page(url):
    logging.info('scraping %s...',url)
    try:
        response=requests.get(url)
        if response.status_code==200:
            return response.text
        logging.error('get invalid status code %s while scraping %s',response.status_code,url)
    except requests.RequestException:
        logging.error('error occurred while scraping %s',url,exc_info=True)
#列表页爬取
def scrape_index(page):
    index_url=f'{BASE_URL}/page/{page}'
    return scrape_page(index_url)
#解析列表页
def parse_index(html):
    doc=pq(html)
    links=doc('.el-card .name')
    for link in links.items():
        href=link.attr('href')
        detail_url=urljoin(BASE_URL,href)
        logging.info('get detail url %s',detail_url)
        yield detail_url
#串联调用
def main():
    for page in range(1,TOTAL_PAGE+1):
        index_html=scrape_index(page)
        detail_urls=parse_index(index_html)
        logging.info('detail urls %s',list(detail_urls))
if __name__=='__main__':
    main()