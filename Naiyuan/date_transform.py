import datetime
import pandas as pd
import os
import docx
import xlrd
from docx import Document
from xlutils.copy import copy
import openpyxl
from openpyxl import load_workbook
import re

#example
'''
data=pd.read_excel(r'./工作簿1.xlsx')
col=list(data.columns)
print(col)
col_new=[]
def date(dates):
    delta=datetime.timedelta(days=dates)
    today=datetime.datetime.strptime('1899-12-30','%Y-%m-%d')+delta
    return datetime.datetime.strftime(today,'%Y-%m-%d')
for x in range(len(col[1:4])):
    col_date=date(col[1:4][x])
    col_new.append(col_date)
col[1:4]=col_new
data.columns=col
print(col)
'''

path='F:\江苏省研究生教育教学改革研究与实践课题'
os.chdir(path)
workbook=xlrd.open_workbook(r'江苏省重点教材立项建设名单（项目版）.xls')
sheet=workbook.sheet_by_index(0)
row_count=sheet.nrows
col_count=sheet.ncols
# print(row_count,col_count)
date=[]
for i in range(2,row_count):
    date.append(sheet.cell(i,13).value)

def date_transform(dates):
    delta=datetime.timedelta(days=dates)
    today=datetime.datetime.strptime('1900-1','%Y-%m')+delta
    return datetime.datetime.strftime(today,'%Y-%m')
date_new=[]
number=[x for x in range(50000)]

for i in range(len(date)):
    for j in number:
        if date[i]==j:
            date[i]=date_transform(date[i])
print(date)

