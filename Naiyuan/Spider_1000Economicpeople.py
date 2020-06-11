from requests_html import HTMLSession
session=HTMLSession()
url='https://ideas.repec.org/top/old/1910/top.person.all.html'
r=session.get(url)
print(r.html.text)
print(r.html.links)
