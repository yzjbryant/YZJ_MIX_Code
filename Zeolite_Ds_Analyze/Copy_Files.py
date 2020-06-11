import os

path="F:\Ds_analyze"

Folder_name=[]
with open(os.path.join(path,'filename.txt'),'r') as f:
    for line in f.readlines():
        x=line[0:3]
        Folder_name.append(x)
print(Folder_name)

# with open(os.path.join(path,'genbox.sh'),'w') as f:
#     str1="#/bin/bash\n"
#     str2="\n"
#     f.write(str1)
#     f.write(str2)
#     for i in range(len(Folder_name)):
#         str3="cd %s\n"%Folder_name[i]
#         str4="mkdir 1\n"\
#              "mkdir 2\n"\
#              "mkdir 3\n"\
#              "mkdir 4\n"\
#              "cd ../\n"
#         str5="\n"
#         f.write(str3)
#         f.write(str4)
#         f.write(str5)

with open(os.path.join(path,'genbox.sh'),'w') as f:
    str1="#/bin/bash\n"
    str2="\n"
    f.write(str1)
    f.write(str2)
    for i in range(len(Folder_name)):
        str4="cd %s/2\n"%Folder_name[i]
        str9="cp ../1/box.g96 .\n"\
             "cd ../3\n" \
             "cp ../1/box.g96 .\n"\
             "cd ../4\n" \
             "cp ../1/box.g96 .\n" \
             "cd ../..\n"
            # str8="cp ../../../../zeolite_CO2/box/%s/1/%s.g96 .\n"%(Folder_name[i],Folder_name[i])
        # str5="genbox_mpi -cp %s.g96 -ci ../../../g96/CH4_TraPPE.g96 -nmol 1200 -o box.g96\n"%Folder_name[i]

        # str6="cd ../..\n"
        str7="\n"
        f.write(str4)
        f.write(str9)
        # f.write(str5)
        # f.write(str6)
        f.write(str7)