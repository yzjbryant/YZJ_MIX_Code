import os
import pandas as pd

pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 6000)
pd.set_option('display.width', 1000)


path2='F:\MOFS_Extra_Data'
data=pd.read_excel('F:\MOFS_Extra_Data\pNature_MOFs.xlsx')
data=data.sort_values(by='Materials Name')


import sys
class Logger(object):
    def __init__(self,fileN="Default.log"):
        self.terminal=sys.stdout
        self.log=open(fileN,"w")
    def write(self,data):
        self.terminal.write(data)
        self.log.write(data)
        self.flush()
    def flush(self):
        self.log.flush()
sys.stdout=Logger("log_file.txt")
print(data)

# Element=['Tong','Zn','C','H','Br','Lu','I','N','O','F','S'] #Cu-Tong,Cl-Lu
# Element2=[] #Number
# Tong=[]
# Zn=[]
# C=[]
# H=[]
# Br=[]
# Lu=[]
# I=[]
# N=[]
# O=[]
# F=[]
# S=[]
#
# with open(os.path.join(path2,'All_Element.txt'),'r') as f:
#     Element3 = ['0' for index in range(11)]  # Number + 0 + ... +0
#     for line in f.readlines():
#         temp=line.split() #Number
#         temp1=line.split() #Element+Number
#         for j in range(len(temp)):
#             for i in range(len(Element)):
#                 temp[j] = temp[j].lstrip(Element[i])
#             Element2.append(temp[j])
#         for j in range(len(temp)):
#             for i in range(len(Element)):
#                 if Element[i] in temp1[j]:
#                     Element3[i]=temp[j]
#         # print(Element3)
#         Tong.append('%s\n'%Element3[0])
#         with open(os.path.join(path2, '1111.txt'), 'a') as f:
#             f.write('%s %s %s %s %s %s %s %s %s %s %s\n'%(Element3[0],Element3[1],Element3[2],Element3[3],
#                                                     Element3[4], Element3[5],Element3[6],Element3[7],
#                                                     Element3[8], Element3[9],Element3[10]))
#     Element2=[]
#     Element3=['0' for index in range(11)]
#
# print(Tong)



