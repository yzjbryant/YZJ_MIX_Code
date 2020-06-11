import os

path="F:\Ds_analyze"

Folder_name=[]
with open(os.path.join(path,'filename.txt'),'r') as f:
    for line in f.readlines():
        x=line[0:3]
        Folder_name.append(x)
print(Folder_name)

with open(os.path.join(path,'genbox.sh'),'w') as f:
    str1="#/bin/bash\n"
    str2="\n"
    # str3="cd /home/yinzj/zeolite/box/   /1\n"
    f.write(str1)
    f.write(str2)
    # f.write(str3)
    for i in range(len(Folder_name)):
        str4="cd %s/2\n"%Folder_name[i]
        str8="cp ../1/gibbs_gromacs_fl.pbs ../1/gromacs.mdp ../1/gromacs.top .\n"\
             "cd ../3\n"\
             "cp ../1/gibbs_gromacs_fl.pbs ../1/gromacs.mdp ../1/gromacs.top .\n"\
             "cd ../4\n" \
             "cp ../1/gibbs_gromacs_fl.pbs ../1/gromacs.mdp ../1/gromacs.top .\n"
        # str5="genbox_mpi -cp %s.g96 -ci ../../../g96/co2_EPM2.g96 -nmol 200 -o box.g96\n"%Folder_name[i]
        str6="cd ../..\n"
        str7="\n"


        f.write(str4)
        f.write(str8)
        # f.write(str5)
        f.write(str6)
        f.write(str7)