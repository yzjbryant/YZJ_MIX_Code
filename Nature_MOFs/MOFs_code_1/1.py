# import pymatgen as mg
# import os
#
# path='F:\MOFS_Extra_Data\est'
#
# a=[]
# structure=mg.Structure.from_file(os.path.join(path,'str_m1_o1_o1_pcu_sym.1.cif'))
# a.append(structure)
#
# doc=open('out.txt','w')
# print(structure,file=doc)

'''
Full_Formula=[]
for i in range(len(Cif_name)):
    structure=mg.Structure.from_file(os.path.join(path,'%s'%Cif_name[i]))
    doc=open('out.txt','w')
    print(structure,file=doc)
    with open(os.path.join(path2,'out.txt'),'r') as f:
        for line in f.readlines()[0:1]:
            x=line
            Full_Formula.append(x)
        f.close()
    print(Full_Formula)
with open(os.path.join(path2,'All.txt'),'w') as f:
    for i in range(len(Full_Formula)):
        f.write(Full_Formula[i])
'''
