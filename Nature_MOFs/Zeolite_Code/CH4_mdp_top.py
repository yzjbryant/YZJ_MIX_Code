import os
import re
from shutil import copyfile

path="F:\Ds_analyze\PDB_itp_g96"

###########################################获取pdb文件名,不要后缀,用于写itp和g96的材料文件名##############
def GetVideoName(dir):
    listName = []
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.g96':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName
file_without_behind_pdb_origin = GetVideoName("F:\Ds_analyze\PDB_itp_g96")



my_file='F:\Ds_analyze\PDB_itp_g96\gibbs_gromacs_fl.pbs'
if os.path.exists(my_file):
    os.remove(my_file)
else:
    print("No file")

my_file='F:\Ds_analyze\PDB_itp_g96\gromacs.mdp'
if os.path.exists(my_file):
    os.remove(my_file)
else:
    print("No file")

my_file='F:\Ds_analyze\PDB_itp_g96\gromacs.top'
if os.path.exists(my_file):
    os.remove(my_file)
else:
    print("No file")



###########################################获取pdb文件名,不要后缀,用于写itp和g96的材料文件名##############
def GetVideoName(dir):
    listName = []
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.g96':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName
file_without_behind_pdb = GetVideoName("F:\Ds_analyze\PDB_itp_g96")

#复制文件
copyfile('F:\Ds_analyze\CH4\gibbs_gromacs_fl.pbs','F:\Ds_analyze\PDB_itp_g96\gibbs_gromacs_fl.pbs')
copyfile('F:\Ds_analyze\CH4\gromacs.mdp','F:\Ds_analyze\PDB_itp_g96\gromacs.mdp')
copyfile('F:\Ds_analyze\CH4\gromacs.top','F:\Ds_analyze\PDB_itp_g96\gromacs.top')
# copyfile('F:\Ds_analyze\CH4\%s.g96'%file_without_behind_pdb[0],'F:\Ds_analyze\PDB_itp_g96\%s.g96'%file_without_behind_pdb[0])




# ###########################################获取pdb文件名,不要后缀,用于写itp和g96的材料文件名##############
# def GetVideoName(dir):
#     listName = []
#     for fileName in os.listdir(dir):
#         if os.path.splitext(fileName)[1] == '.g96':
#             fileName = os.path.splitext(fileName)[0]
#             listName.append(fileName)
#     return listName
# file_without_behind_pdb = GetVideoName("F:\Ds_analyze\PDB_itp_g96")

############################################改写gromacs.top文件###################################################

with open(os.path.join(path,'gromacs.top'), 'r') as f:
    top_filename=[]
    for line in f.readlines()[12:13]:
       x=line[18:21]
       top_filename.append(x)

with open(os.path.join(path,'gromacs.top'), 'r') as f:
    alllines=f.readlines()
    f.close()

with open(os.path.join(path,'gromacs.top'),'w+') as f:
    for eachline in alllines:
        a=re.sub('%s'%top_filename[0],'%s'%file_without_behind_pdb[0],eachline)
        f.writelines(a)
    f.close()


############################################改写gromacs.mdp文件###################################################

with open(os.path.join(path,'gromacs.mdp'), 'r') as f:
    mdp_filename=[]
    for line in f.readlines()[47:48]:
       x=line[31:34]
       mdp_filename.append(x)

with open(os.path.join(path,'gromacs.mdp'), 'r') as f:
    alllines=f.readlines()
    f.close()

with open(os.path.join(path,'gromacs.mdp'),'w+') as f:
    for eachline in alllines:
        a=re.sub('%s'%mdp_filename[0],'%s'%file_without_behind_pdb[0],eachline)
        f.writelines(a)
    f.close()

my_file='F:\Ds_analyze\PDB_itp_g96\%s.g96'%file_without_behind_pdb_origin[0]
if os.path.exists(my_file):
    os.remove(my_file)
else:
    print("No file")