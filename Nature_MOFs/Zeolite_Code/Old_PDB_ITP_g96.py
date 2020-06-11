import os
import numpy as np
import math

##############################################给出pdb&cif的路径######################################
path="F:\Ds_analyze\PDB_itp_g96"
files_name=os.listdir(path)

##############################################获取文件夹下文件名######################################
x=os.listdir(path)
print(x)

###########################################获取pdb文件名,不要后缀,用于写itp和g96的材料文件名############
def GetVideoName(dir):
    listName = []
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.pdb':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName

file_without_behind_pdb = GetVideoName("F:\Ds_analyze\PDB_itp_g96")
print(file_without_behind_pdb)

#cif---[0]
#pdb---[1]



############这部分是读.cif文件##############
#读材料名

with open(os.path.join(path,"%s"%files_name[0]), 'r') as f:
    CIF_name=[]
    for line in f.readlines()[0:1]:
        x=line[5:8]
        CIF_name.append(x)

#先读cif中单个cell的三边长度（单位：埃）
with open(os.path.join(path,"%s"%files_name[0]),'r') as f1:
    length_onecell = []
    for line in f1.readlines()[14:15]:
        x = line[17:26]
        length_onecell.append(x)
with open(os.path.join(path,"%s"%files_name[0]), 'r') as f1:
    for line in f1.readlines()[15:16]:
        z = line[17:26]
        length_onecell.append(z)
with open(os.path.join(path,"%s"%files_name[0]), 'r') as f1:
    for line in f1.readlines()[16:17]:
        u = line[17:26]
        length_onecell.append(u)

#再读cif中单个cell的三个degree（单位：角度）
with open(os.path.join(path,"%s"%files_name[0]),'r') as f1:
    Degree_onecell = []
    for line in f1.readlines()[17:18]:
        x = line[17:28]
        Degree_onecell.append(x)
with open(os.path.join(path,"%s"%files_name[0]), 'r') as f1:
    for line in f1.readlines()[18:19]:
        z = line[17:28]
        Degree_onecell.append(z)
with open(os.path.join(path,"%s"%files_name[0]), 'r') as f1:
    for line in f1.readlines()[19:20]:
        u = line[17:28]
        Degree_onecell.append(u)













#取PDB中超晶胞三边总长 （单位：埃）
with open(os.path.join(path,"%s"%files_name[1]), 'r') as f2:
    length_total_cell = []
    for line in f2.readlines()[2:3]:
        x = line[6:15]
        length_total_cell.append(x)
        z = line[15:24]
        length_total_cell.append(z)
        u = line[24:33]
        length_total_cell.append(u)
print(length_onecell)
print(Degree_onecell)
print(length_total_cell)

# #去除单引号
length_onecell_eval=list(map(eval,length_onecell))
Degree_onecell_eval=list(map(eval,Degree_onecell))
length_total_cell_eval=list(map(eval,length_total_cell))



#计算插入盒子个数的函数
Number_of_cell=[length_total_cell_eval[i] / length_onecell_eval[i] for i in range(len(length_total_cell_eval))]
Number_of_cell_int = []
for i in range(len(Number_of_cell)):
    z=int(Number_of_cell[i])
    Number_of_cell_int.append(z)

# 写itp文件
with open(os.path.join(path,'%s.itp'%file_without_behind_pdb[0]), 'a') as f3:
        str1=";Molecules are used in GROMACS\n"\
            ";Created by ZhiJianYin at March\n"
        str2=";This is file is for  %s  materials its sells %d * %d * %d  in UFF force field\n" %(CIF_name[0],Number_of_cell_int[0],Number_of_cell_int[1],Number_of_cell_int[2])
        str3=";a = %d, beta = %d, gamma = %d (degree)\n" %(Degree_onecell_eval[0],Degree_onecell_eval[1],Degree_onecell_eval[2])
        str4=";\n"\
             ";Reference ??\n"\
             ";\n"\
             "\n"\
             "[ moleculetype ]\n"\
             "; name  nrexcl\n"
        str5="%s    3\n"% CIF_name[0]
        str6="\n"\
             "[ atoms ]\n"\
             "; nr      type    resnr    residu    atom   cgnr    charge\n"
        f3.write(str2)
        f3.write(str1)
        f3.write(str3)
        f3.write(str4)
        f3.write(str5)
        f3.write(str6)

#读PDB中总atom的总原子数
with open(os.path.join(path,"%s"%files_name[1]),'r') as fp:
    atom_number=0
    for i in fp.readlines():
        if i.startswith("ATOM"):
            atom_number +=1


#追加itp中nr,cgnr序号
with open(os.path.join(path,"%s"%files_name[1]),'r') as f4:
    number_of_nr_cgnr = []
    for line in f4.readlines()[9:atom_number+9]:
        x = line[4:11]
        number_of_nr_cgnr.append(x)

#itp中residu为r
#itp中resnr为1


#读PDB中atom名
with open(os.path.join(path,"%s"%files_name[1]),'r') as f5:
    atom_name=[]
    for line in f5.readlines()[9:atom_number+9]:
        x=line[76:78]
        atom_name.append(x)

#追加itp中type和atom名
type_atom_name=[x+'_z' for x in atom_name]

#定义atom电荷
charge=[' 2.05000','-1.02500']
charge_atom=[]
for i in atom_name:
    if i =='Si':
        charge_atom.append(charge[0])
    else:
        charge_atom.append(charge[1])



#开始追加itp
with open(os.path.join(path,'%s.itp'%file_without_behind_pdb[0]),'a') as f6:
    for line in range(len(type_atom_name)):
        str8="%s       %s     1    %s       %s  %s       %s\n"%(number_of_nr_cgnr[line],type_atom_name[line],CIF_name[0],type_atom_name[line],number_of_nr_cgnr[line],charge_atom[line])
        f6.write(str8)

#获取pdb坐标文件
with open(os.path.join(path,"%s"%files_name[1]),'r') as f7:
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

#给坐标乘以0.1
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
# print(length_total_cell_eval_0_1)

#定义二维数组
a=[length_total_cell_eval_0_2[0],0,0]
b=[0,0,0]
c=[0,0,0]
Box_vector=np.array([a,b,c])

if ((Degree_onecell_eval[0] != 90.0 or Degree_onecell_eval[1] !=90.0 or Degree_onecell_eval[2] != 90.0)):
    if (Degree_onecell_eval[0] != 90.0):
        cosa=math.cos(math.radians(Degree_onecell_eval[0]))
    else:
        cosa=0

    if (Degree_onecell_eval[1] != 90.0):
        cosb=math.cos(math.radians(Degree_onecell_eval[1]))
    else:
        cosb=0

    if (Degree_onecell_eval[2] != 90.0):
        cosg=math.cos(math.radians(Degree_onecell_eval[2]))
        sing=math.sin(math.radians(Degree_onecell_eval[2]))
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
# print(Coordinate_z_eval_0_1_add_0)

Coordinate_x_eval_0_1_add_0_eval=list(map(eval,Coordinate_x_eval_0_1_add_0))
Coordinate_y_eval_0_1_add_0_eval=list(map(eval,Coordinate_y_eval_0_1_add_0))
Coordinate_z_eval_0_1_add_0_eval=list(map(eval,Coordinate_z_eval_0_1_add_0))
# print(Coordinate_x_eval_0_1_add_0_eval)
nr=[1]
#开始写g96文件
with open(os.path.join(path,'%s.g96'%file_without_behind_pdb[0]),'a') as f8:
    str1="TITLE\n"
    str2="This molecular %s file was created by ZhiJianYin\n" %CIF_name[0]
    str3="END\n"
    str4="POSITION\n"
    f8.write(str1)
    f8.write(str2)
    f8.write(str3)
    f8.write(str4)

    for line in range(len(type_atom_name)):
        str5="%5d %s %s   %s%s%s%s\n" %(nr[0],CIF_name[0],type_atom_name[line],number_of_nr_cgnr[line],Coordinate_x_eval_0_1_add_0[line],Coordinate_y_eval_0_1_add_0[line],Coordinate_z_eval_0_1_add_0[line])
        f8.write(str5)
    str6="END\n"\
        "BOX\n"
    str7="%.9f %.9f %.9f %9.f %.9f %.9f %9.f %.9f %.9f\n"%(Box_vector[0,0],Box_vector[1,1],Box_vector[2,2],Box_vector[0,1],Box_vector[0,2],Box_vector[1,0],Box_vector[1,2],Box_vector[2,0],Box_vector[2,1])
    str8="END\n"
    f8.write(str6)
    f8.write(str7)
    f8.write(str8)