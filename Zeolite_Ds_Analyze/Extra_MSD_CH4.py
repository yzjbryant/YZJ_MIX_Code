import os
import sys


folder_name=['ABW']

folder_next=['1','2','3','4']

ds=[]
ds_line_x=[]
ds_line_y=[]
for i in folder_name:
    os.chdir('%s'%i)

    top_number=[]
    for j in folder_next:
        path='%s'%j

        with open(os.path.join(path,'msd.xvg'),'r') as f:
            for line in f.readlines()[14:15]:
                x=line[17:51]
                ds.append(x)

        with open(os.path.join(path,'msd.xvg'),'r') as f:
            for line in f.readlines()[15:216]:
                x=line[0:10]
                y=line[10:22]
                ds_line_x.append(x)

        with open(os.path.join(path,'msd.xvg'),'r') as f:
            for line in f.readlines()[15:216]:
                x=line[10:22]
                ds_line_y.append(x)

        with open(os.path.join(path,'gromacs.top'),'r') as f:
            for line in f.readlines()[23:24]:
                x=line[24:28]
                y=x.strip()
                top_number.append(y)


    with open('data.mdp', 'a') as f:

        str2='%s\n'%folder_name[0]
        f.write(str2)
        for line in range(len(ds)):
            str = '%s\n' % ds[line]
            f.write(str)
        for line in range(len(ds_line_y)):
            str1 = " %s %s\n" % (ds_line_x[line], ds_line_y[line])
            f.write(str1)
        for line in range(len(top_number)):
            str5='%s\n'%top_number[line]
            f.write(str5)
    rootpath = os.path.dirname(sys.path[0])
    os.chdir(rootpath)










