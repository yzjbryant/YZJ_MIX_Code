# print("This is yinzhijan ")

#weatherman.py
# import report
# description = report.get_description()
# print("sdadadada",description)

# import report as mr
# description = mr.get_description()
# print("sdadadada",description)

#导入模块一部分
# from report import get_description as do_it
# description = do_it()
# print("dsad",description)

#写列表的行
# import csv
# villains=[
#     ['Doctor','No'],
#     ['dada','dasda'],
#     ['adad','dadas'],
# ]
# with open('villains','wt') as fout:
#     csvout=csv.writer(fout)
#     csvout.writerows(villains)
#
# import csv
# with open('villains','rt') as fin:
#     # cin=csv.reader(fin)
#     cin = csv.DictReader(fin,fieldnames=['first','last'])
#     villains=[row for row in cin]
# print(villains)

# import csv
# villains=[
#     {'first':'sdsds','last':'sdsdsd'},
#     {'first':'dasdada','last':'sadasd'},
# ]
# with open('vi','wt') as fout:
#     cout=csv.DictWriter(fout,['first','last'])
#     cout.writeheader()
#     cout.writerows(villains)

# #读文件

# test1='adsada'
# with open('ad','wt') as fin:
#     fin.write(test1)
#
# with open('ad','rt') as infle:
#     test2=infle.read()
# len(test2)
# test1==test2

# from datetime import date
# now = date.today()
# now_str=now,isoformat()
# with open('today','wt') as output:
#     print(now_str,file=output)
#
# with open('today','rt') as input:
#     today_string=input.read()
#
# fmt='%Y-%m-%d\n'
# datetime.strptime(today_string,fmt)
#
# import os #子目录
# os.listdir('.')
#
# import os #父目录
# os.listdir('..')

# import multiprocessing
#
# def now(seconds):
#     from datetime import datetime
#     from time import sleep
#     sleep(seconds)
#     print('wait',seconds,'seconds,time is',datetime.utcnow())
#
# if __name__ == '__main__':
#     import random
#     for n in range(3):
#         seconds=random.random()
#         proc=multiprocessing.Process(target=now,args=(seconds,))
#         proc.start()

my_day=date(1982,8,14)
my_day
my_day.weekday()
my_day.isoweekday()


from datetime import timedelta
party_day=my_day + timedelta(days=10000)
party_day


