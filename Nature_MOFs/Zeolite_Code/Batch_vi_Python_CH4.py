import os
from shutil import copyfile
import re

path1='F:\Ds_analyze\PDB_itp_g96'#gibbs
path2='F:\Ds_analyze'            #python
path3='F:\Ds_analyze\Py'         #写python
path4='F:\Ds_analyze\Gibbs'      #写gibbs

Folder_name=[]
with open(os.path.join(path2,'filename.txt'),'r') as f:
    for line in f.readlines():
        x=line[0:3]
        Folder_name.append(x)
# print(Folder_name)


for i in range(len(Folder_name)):
    # copyfile('F:\Ds_analyze\PDB_itp_g96\gibbs_gromacs_fl2.pbs','F:\Ds_analyze\Gibbs\gibbs_gromacs_fl_%s.pbs'%Folder_name[i])
    copyfile('F:\Ds_analyze\Cut_box_vi_top_CH4.py', 'F:\Ds_analyze\Py\Cut_box_vi_top_CH4_%s.py'%Folder_name[i])


# ####Gibbs
# for i in range(len(Folder_name)):
#     fp=open(os.path.join(path4,'gibbs_gromacs_fl_%s.pbs'%Folder_name[i]))
#     s=fp.read()
#     fp.close()
#     a=s.split('\n')
#     a.insert(61,'python Cut_box_vi_top_CH4_%s.py'%Folder_name[i])
#     s='\n'.join(a)
#     fp=open(os.path.join(path4,'gibbs_gromacs_fl_%s.pbs'%Folder_name[i]),'w')
#     fp.write(s)
#     fp.close()

####Py
for i in range(len(Folder_name)):

    with open(os.path.join(path3, 'Cut_box_vi_top_CH4_%s.py'%Folder_name[i]), 'r') as f:
        alllines = f.readlines()

    with open(os.path.join(path3, 'Cut_box_vi_top_CH4_%s.py'%Folder_name[i]), 'w+') as f:
        for eachline in alllines:
            a = re.sub('AEN', '%s' %Folder_name[i], eachline)
            f.writelines(a)
        f.close()

with open(os.path.join(path3,'genbox.sh'),'w') as f:
    str1="#/bin/bash\n"
    str2="\n"

    f.write(str1)
    f.write(str2)

    for i in range(len(Folder_name)):
        # str3="python Cut_box_vi_top_CH4_%s.py\n"%Folder_name[i]
        str3="rm Cut_box_vi_top_CH4_%s.py\n"%Folder_name[i]
        str4="\n"
        f.write(str3)
        f.write(str4)
