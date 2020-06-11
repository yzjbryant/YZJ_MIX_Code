import os
import re
from shutil import copyfile

path='F:\Ds_analyze'
path1='F:\Ds_analyze\PDB_itp_g96'




Material_name=['ITV']





#删文件
my_file='F:\Ds_analyze\gibbs_gromacs_fl2.pbs'
if os.path.exists(my_file):
    os.remove(my_file)
else:
    print("No file")

#复制文件
copyfile('F:\Ds_analyze\PDB_itp_g96\gibbs_gromacs_fl2.pbs','F:\Ds_analyze\gibbs_gromacs_fl2.pbs')

#写文件
fp=open('gibbs_gromacs_fl2.pbs')
s=fp.read()
fp.close()
a=s.split('\n')

number=[]
for i in range(50,70):
    x=i*100
    number.append(x)
print(number)

for i in range(len(number)):

    # a.insert(61,'genbox_mpi -cp /home/yinzj/zeolite/box/%s/1/%s.g96 -ci /home/yinzj/zeolite/g96/co2_EPM2.g96 -nmol %s -o box.g96\n'
    #          %(Material_name[0],Material_name[0],number[i]))

    a.insert(61, 'genbox_mpi -cp /home/yinzj/zeolite/box/%s/1/%s.g96 -ci /home/yinzj/zeolite/g96/CH4_TraPPE.g96 -nmol %s -o box.g96\n'
             %(Material_name[0], Material_name[0],number[i]))
    s='\n'.join(a)
fp=open('gibbs_gromacs_fl2.pbs','w')
fp.write(s)
fp.close()