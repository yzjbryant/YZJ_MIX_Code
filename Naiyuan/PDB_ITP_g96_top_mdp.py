import os
import numpy as np
import math
import re
path="F:\Ds_analyze\PDB_itp_g96"




##############################################获取文件夹下文件名########################################
files_name=os.listdir(path)
# print(files_name)




###########################################获取pdb文件名,不要后缀,用于写itp和g96的材料文件名##############
def GetVideoName(dir):
    listName = []
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.pdb':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName
file_without_behind_pdb = GetVideoName("F:\Ds_analyze\PDB_itp_g96")




#########################################读PDB中三边总长###############################################
with open(os.path.join(path, "%s.pdb" % file_without_behind_pdb[0]), 'r') as f:
    length_total_cell = []
    for line in f.readlines()[2:3]:
        x = line[6:15]
        length_total_cell.append(x)
        z = line[15:24]
        length_total_cell.append(z)
        u = line[24:33]
        length_total_cell.append(u)
        # 去除列表中的单引号
length_total_cell_eval = list(map(eval, length_total_cell))
print(length_total_cell_eval)



#####################如果文件夹中存在cif文件，把单晶胞中的a,b,c三边长读取##################################
if '%s.cif'%file_without_behind_pdb[0] in files_name:
    with open(os.path.join(path, "%s.cif" % file_without_behind_pdb[0]), 'r') as f:
        length_onecell = []
        for line in f.readlines()[14:15]:
            x = line[17:26]
            length_onecell.append(x)
    with open(os.path.join(path, "%s.cif" % file_without_behind_pdb[0]), 'r') as f:
        for line in f.readlines()[15:16]:
            z = line[17:26]
            length_onecell.append(z)
    with open(os.path.join(path, "%s.cif" % file_without_behind_pdb[0]), 'r') as f:
        for line in f.readlines()[16:17]:
            u = line[17:26]
            length_onecell.append(u)
            # 去除列表中的单引号
            length_onecell_eval=list(map(eval,length_onecell))




############读取插入晶胞个数#############################################################################
            Number_of_cell = [length_total_cell_eval[i] / length_onecell_eval[i] for i in
                              range(len(length_total_cell_eval))]
            Number_of_cell_int = []
            for i in range(len(Number_of_cell)):
                z = int(Number_of_cell[i])
                Number_of_cell_int.append(z)
            print(Number_of_cell_int)
else:
    print("No cif")





########################################读PDB中三个degree###############################################
with open(os.path.join(path, "%s.pdb" % file_without_behind_pdb[0]), 'r') as f:
    Degree_supercell = []
    for line in f.readlines()[2:3]:
        x = line[33:41]
        Degree_supercell.append(x)
        z = line[41:47]
        Degree_supercell.append(z)
        u = line[48:55]
        Degree_supercell.append(u)
# 去除列表中的单引号
Degree_supercell_eval = list(map(eval, Degree_supercell))
print(Degree_supercell_eval)



####################写itp文件###########################################################################
with open(os.path.join(path,'%s.itp'%file_without_behind_pdb[0]), 'a') as f3:
        str1=";Molecules are used in GROMACS\n"\
            ";Created by ZhiJianYin at March\n"
        if '%s.cif' % file_without_behind_pdb[0] in files_name:
            str2=";This is file is for  %s  materials its sells %d * %d * %d  in UFF force field\n" %(file_without_behind_pdb[0],Number_of_cell_int[0],Number_of_cell_int[1],Number_of_cell_int[2])
        else:
            print("No cif")
        str3=";a = %d, beta = %d, gamma = %d (degree)\n" %(Degree_supercell_eval[0],Degree_supercell_eval[1],Degree_supercell_eval[2])
        str4=";\n"\
             ";Reference ??\n"\
             ";\n"\
             "\n"\
             "[ moleculetype ]\n"\
             "; name  nrexcl\n"
        str5="%s    3\n"% file_without_behind_pdb[0]
        str6="\n"\
             "[ atoms ]\n"\
             "; nr      type    resnr    residu    atom   cgnr    charge\n"
        if '%s.cif' % file_without_behind_pdb[0] in files_name:
            f3.write(str2)
        else:
            print("No cif")
        f3.write(str1)
        f3.write(str3)
        f3.write(str4)
        f3.write(str5)
        f3.write(str6)

#################################################读PDB中总atom的总原子数##################################
with open(os.path.join(path,"%s.pdb"%file_without_behind_pdb[0]),'r') as fp:
    atom_number=0
    for i in fp.readlines():
        if i.startswith("ATOM"):
            atom_number +=1

####################################################追加itp中nr,cgnr序号#################################
with open(os.path.join(path,"%s.pdb"%file_without_behind_pdb[0]),'r') as f4:
    number_of_nr_cgnr = []
    for line in f4.readlines()[9:atom_number+9]:
        x = line[4:11]
        number_of_nr_cgnr.append(x)

###############################################itp中resnr为1#############################################

################################################读PDB中atom名############################################
with open(os.path.join(path,"%s.pdb"%file_without_behind_pdb[0]),'r') as f5:
    atom_name=[]
    for line in f5.readlines()[9:atom_number+9]:
        x=line[76:78]
        atom_name.append(x)

###############################################追加itp中type和atom名#####################################
type_atom_name=[x+'_z' for x in atom_name]

###############################################定义atom电荷##############################################
charge=[' 2.05000','-1.02500']
charge_atom=[]
for i in atom_name:
    if i =='Si':
        charge_atom.append(charge[0])
    else:
        charge_atom.append(charge[1])

###############################################开始追加itp###############################################
with open(os.path.join(path,'%s.itp'%file_without_behind_pdb[0]),'a') as f6:
    for line in range(len(type_atom_name)):
        str8="%s       %s     1    %s       %s  %s       %s\n"%(number_of_nr_cgnr[line],type_atom_name[line],file_without_behind_pdb[0],type_atom_name[line],number_of_nr_cgnr[line],charge_atom[line])
        f6.write(str8)

#####################################################获取pdb坐标文件######################################
with open(os.path.join(path,'%s.pdb'%file_without_behind_pdb[0]),'r') as f7:
    Coordinate_x = []
    Coordinate_y = []
    Coordinate_z = []
    for line in f7.readlines()[9:atom_number+9]:
        x=line[28:38]
        y=line[38:46]
        z=line[46:54]

        Coordinate_x.append(x)
        Coordinate_y.append(y)
        Coordinate_z.append(z)

#去除单引号
Coordinate_x_eval=list(map(eval,Coordinate_x))
Coordinate_y_eval=list(map(eval,Coordinate_y))
Coordinate_z_eval=list(map(eval,Coordinate_z))

#给pdb三个坐标乘以0.1
def multiply(x):
    return x*0.1
def function(p,y):
    res=[]
    for i in y:
        list=p(i)
        res.append(list)
    return res
Coordinate_x_eval_0_1=function(multiply,Coordinate_x_eval)
Coordinate_y_eval_0_1=function(multiply,Coordinate_y_eval)
Coordinate_z_eval_0_1=function(multiply,Coordinate_z_eval)
length_total_cell_eval_0_1=function(multiply,length_total_cell_eval)



#给坐标乘以0.1
def multiply_1(x):
    return x*0.1
def function_1(p,y):
    res=[]
    for i in y:
        list=p(i)
        res.append(list)
    return res
length_total_cell_eval_0_2=function_1(multiply_1,length_total_cell_eval)

###################################################定义三维数组################################################
a=[length_total_cell_eval_0_2[0],0,0]
b=[0,0,0]
c=[0,0,0]
Box_vector=np.array([a,b,c])

if ((Degree_supercell_eval[0] != 90.0 or Degree_supercell_eval[1] !=90.0 or Degree_supercell_eval[2] != 90.0)):
    if (Degree_supercell_eval[0] != 90.0):
        cosa=math.cos(math.radians(Degree_supercell_eval[0]))
    else:
        cosa=0

    if (Degree_supercell_eval[1] != 90.0):
        cosb=math.cos(math.radians(Degree_supercell_eval[1]))
    else:
        cosb=0

    if (Degree_supercell_eval[2] != 90.0):
        cosg=math.cos(math.radians(Degree_supercell_eval[2]))
        sing=math.sin(math.radians(Degree_supercell_eval[2]))
    else:
        cosg=0
        sing=1
    Box_vector[1,0]=length_total_cell_eval_0_2[1]*cosg
    Box_vector[1,1]=length_total_cell_eval_0_2[1]*sing
    Box_vector[2,0]=length_total_cell_eval_0_2[2]*cosb
    Box_vector[2,1]=length_total_cell_eval_0_2[2]*(cosa-cosb*cosg)/sing
    Box_vector[2,2]=math.sqrt(length_total_cell_eval_0_2[2]*length_total_cell_eval_0_2[2]-
                              Box_vector[2,0]*Box_vector[2,0]-Box_vector[2,1]*Box_vector[2,1])
else:
    Box_vector[1,1]=length_total_cell_eval_0_2[1]
    Box_vector[2,2]=length_total_cell_eval_0_2[2]

# print(Box_vector)

Coordinate_x_eval_0_1_add_0=[]
for i in Coordinate_x_eval_0_1:
    s="%15.9f"%i
    Coordinate_x_eval_0_1_add_0.append(s)


Coordinate_y_eval_0_1_add_0=[]
for i in Coordinate_y_eval_0_1:
    s="%15.9f"%i
    Coordinate_y_eval_0_1_add_0.append(s)

Coordinate_z_eval_0_1_add_0=[]
for i in Coordinate_z_eval_0_1:
    s="%15.9f"%i
    Coordinate_z_eval_0_1_add_0.append(s)


Coordinate_x_eval_0_1_add_0_eval=list(map(eval,Coordinate_x_eval_0_1_add_0))
Coordinate_y_eval_0_1_add_0_eval=list(map(eval,Coordinate_y_eval_0_1_add_0))
Coordinate_z_eval_0_1_add_0_eval=list(map(eval,Coordinate_z_eval_0_1_add_0))

nr=[1]

###########################################################开始写g96文件#########################################
with open(os.path.join(path,'%s.g96'%file_without_behind_pdb[0]),'a') as f8:
    str1="TITLE\n"
    str2="This molecular %s file was created by ZhiJianYin\n" %file_without_behind_pdb[0]
    str3="END\n"
    str4="POSITION\n"
    f8.write(str1)
    f8.write(str2)
    f8.write(str3)
    f8.write(str4)

    for line in range(len(type_atom_name)):
        str5="%5d %s %s   %s%s%s%s\n" %(nr[0],file_without_behind_pdb[0],type_atom_name[line],number_of_nr_cgnr[line],Coordinate_x_eval_0_1_add_0[line],Coordinate_y_eval_0_1_add_0[line],Coordinate_z_eval_0_1_add_0[line])
        f8.write(str5)
    str6="END\n"\
        "BOX\n"
    str7="%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n"%(Box_vector[0,0],Box_vector[1,1],Box_vector[2,2],Box_vector[0,1],Box_vector[0,2],Box_vector[1,0],Box_vector[1,2],Box_vector[2,0],Box_vector[2,1])
    str8="END\n"
    f8.write(str6)
    f8.write(str7)
    f8.write(str8)


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



with open(os.path.join(path,'gromacs.mdp'),'w+') as f:
    for eachline in alllines:
        a=re.sub('%s'%mdp_filename[0],'%s'%file_without_behind_pdb[0],eachline)
        f.writelines(a)
    f.close()