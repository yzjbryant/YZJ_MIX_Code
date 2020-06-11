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

    f.write(str1)
    f.write(str2)

    for i in range(len(Folder_name)):
        str3="cd %s/1\n"%Folder_name[i]
        # str4='dos2unix gibbs_gromacs_fl2.pbs\n'
        str5="qsub gibbs_gromacs_fl.pbs\n"\
             "cd ../2\n"\
             "qsub gibbs_gromacs_fl.pbs\n"\
             "cd ../3\n"\
             "qsub gibbs_gromacs_fl.pbs\n" \
             "cd ../4\n" \
             "qsub gibbs_gromacs_fl.pbs\n" \
             "cd ../..\n"\
             "\n"
        f.write(str3)
        # f.write(str4)
        f.write(str5)
