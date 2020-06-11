import os
import sys
import re


folder_name=['ITR']
folder_next=['1','2','3','4']



ds=[]
ds_line_x=[]
ds_line_y=[]
Number=[]
Number_atom=[]
for i in folder_name:
    os.chdir('%s'%i)



#########################################
    for j in folder_next[0]:
        path='%s'%j


        count=len(open('box.g96','rU').readlines())
        Number.append(count-8)
        # print(Number)

        with open('box.g96','r') as f:
            count=0
            for line in f.readlines():
                if line.startswith("    1 %s"%folder_name[0]):
                    count+=1
            # print(count)

        Number_atom.append(Number[0]-count)


        Last_Number = [4]
        Update_Last_Number = []
        New_last_number = []
        Second_number = []
        Third_number = []
        All = []

        if (Number_atom[0] - Last_Number[0]) % 3 == 0:
            x = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3
            y = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3 * 2
            Second_number.append(x)
            Third_number.append(y)
            # print(Second_number, Third_number)

            # print(Number_atom[0], Second_number[0], Third_number[0], Last_Number[0])
            All.append(Last_Number[0])
        else:
            z = (Number_atom[0] - Last_Number[0]) % 3
            Update_Last_Number.append(z)
            u = Last_Number[0] + Update_Last_Number[0]
            New_last_number.append(u)
            if (Number_atom[0] - New_last_number[0]) % 3 == 0:
                x = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3
                y = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3 * 2
                Second_number.append(x)
                Third_number.append(y)
                # print(Second_number, Third_number)

                # print(Number_atom[0], Second_number[0], Third_number[0], New_last_number[0])
                All.append(New_last_number[0])

        All.append(Third_number[0])
        All.append(Second_number[0])
        All.append(Number_atom[0])


        with open(os.path.join(path, 'gromacs.top'), 'r') as f:
            alllines = f.readlines()
            f.close()

        with open(os.path.join(path, 'gromacs.top'), 'w+') as fp:
            for eachline in alllines:
                s = re.sub('CH4                 ', 'CH4                     %s'%All[3],eachline)
                fp.writelines(s)





#######################################
    for j in folder_next[1]:
        path='%s'%j

        count=len(open('box.g96','rU').readlines())
        Number.append(count-8)
        # print(Number)

        with open('box.g96','r') as f:
            count=0
            for line in f.readlines():
                if line.startswith("    1 %s"%folder_name[0]):
                    count+=1
            # print(count)

        Number_atom.append(Number[0]-count)


        Last_Number = [4]
        Update_Last_Number = []
        New_last_number = []
        Second_number = []
        Third_number = []
        All = []

        if (Number_atom[0] - Last_Number[0]) % 3 == 0:
            x = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3
            y = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3 * 2
            Second_number.append(x)
            Third_number.append(y)
            # print(Second_number, Third_number)

            # print(Number_atom[0], Second_number[0], Third_number[0], Last_Number[0])
            All.append(Last_Number[0])
        else:
            z = (Number_atom[0] - Last_Number[0]) % 3
            Update_Last_Number.append(z)
            u = Last_Number[0] + Update_Last_Number[0]
            New_last_number.append(u)
            if (Number_atom[0] - New_last_number[0]) % 3 == 0:
                x = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3
                y = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3 * 2
                Second_number.append(x)
                Third_number.append(y)
                # print(Second_number, Third_number)

                # print(Number_atom[0], Second_number[0], Third_number[0], New_last_number[0])
                All.append(New_last_number[0])

        All.append(Third_number[0])
        All.append(Second_number[0])
        All.append(Number_atom[0])


        with open(os.path.join(path, 'gromacs.top'), 'r') as f:
            alllines = f.readlines()
            f.close()

        with open(os.path.join(path, 'gromacs.top'), 'w+') as fp:
            for eachline in alllines:
                s = re.sub('CH4                 ', 'CH4                     %s'%All[2],eachline)
                fp.writelines(s)

        ################################################################################delete_box
        x = [j for j in range(All[2]+2, All[3] + 2)]
        print(x)
        for i in x:
            with open(os.path.join(path, 'box.g96'), 'r') as f:
                lines = f.readlines()

            with open(os.path.join(path, 'box.g96'), 'w') as f:
                for l in lines:
                    if ' %s CH4' % i not in l:
                        f.write(l)




##################################################
    for j in folder_next[2]:
        path='%s'%j


        count=len(open('box.g96','rU').readlines())
        Number.append(count-8)
        # print(Number)

        with open('box.g96','r') as f:
            count=0
            for line in f.readlines():
                if line.startswith("    1 %s"%folder_name[0]):
                    count+=1
            # print(count)

        Number_atom.append(Number[0]-count)


        Last_Number = [4]
        Update_Last_Number = []
        New_last_number = []
        Second_number = []
        Third_number = []
        All = []

        if (Number_atom[0] - Last_Number[0]) % 3 == 0:
            x = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3
            y = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3 * 2
            Second_number.append(x)
            Third_number.append(y)
            # print(Second_number, Third_number)

            # print(Number_atom[0], Second_number[0], Third_number[0], Last_Number[0])
            All.append(Last_Number[0])
        else:
            z = (Number_atom[0] - Last_Number[0]) % 3
            Update_Last_Number.append(z)
            u = Last_Number[0] + Update_Last_Number[0]
            New_last_number.append(u)
            if (Number_atom[0] - New_last_number[0]) % 3 == 0:
                x = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3
                y = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3 * 2
                Second_number.append(x)
                Third_number.append(y)
                # print(Second_number, Third_number)

                # print(Number_atom[0], Second_number[0], Third_number[0], New_last_number[0])
                All.append(New_last_number[0])

        All.append(Third_number[0])
        All.append(Second_number[0])
        All.append(Number_atom[0])


        with open(os.path.join(path, 'gromacs.top'), 'r') as f:
            alllines = f.readlines()
            f.close()

        with open(os.path.join(path, 'gromacs.top'), 'w+') as fp:
            for eachline in alllines:
                s = re.sub('CH4                 ', 'CH4                     %s'%All[1],eachline)
                fp.writelines(s)

        ################################################################################delete_box
        x = [j for j in range(All[1]+2, All[3] + 2)]
        print(x)
        for i in x:
            with open(os.path.join(path, 'box.g96'), 'r') as f:
                lines = f.readlines()

            with open(os.path.join(path, 'box.g96'), 'w') as f:
                for l in lines:
                    if ' %s CH4' % i not in l:
                        f.write(l)



################################################
    for j in folder_next[3]:
        path='%s'%j



        count=len(open('box.g96','rU').readlines())
        Number.append(count-8)
        # print(Number)

        with open('box.g96','r') as f:
            count=0
            for line in f.readlines():
                if line.startswith("    1 %s"%folder_name[0]):
                    count+=1
            # print(count)

        Number_atom.append(Number[0]-count)


        Last_Number = [4]
        Update_Last_Number = []
        New_last_number = []
        Second_number = []
        Third_number = []
        All = []

        if (Number_atom[0] - Last_Number[0]) % 3 == 0:
            x = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3
            y = Number_atom[0] - (Number_atom[0] - Last_Number[0]) / 3 * 2
            Second_number.append(x)
            Third_number.append(y)
            # print(Second_number, Third_number)

            # print(Number_atom[0], Second_number[0], Third_number[0], Last_Number[0])
            All.append(Last_Number[0])
        else:
            z = (Number_atom[0] - Last_Number[0]) % 3
            Update_Last_Number.append(z)
            u = Last_Number[0] + Update_Last_Number[0]
            New_last_number.append(u)
            if (Number_atom[0] - New_last_number[0]) % 3 == 0:
                x = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3
                y = Number_atom[0] - (Number_atom[0] - New_last_number[0]) / 3 * 2
                Second_number.append(x)
                Third_number.append(y)
                # print(Second_number, Third_number)

                # print(Number_atom[0], Second_number[0], Third_number[0], New_last_number[0])
                All.append(New_last_number[0])

        All.append(Third_number[0])
        All.append(Second_number[0])
        All.append(Number_atom[0])


        with open(os.path.join(path, 'gromacs.top'), 'r') as f:
            alllines = f.readlines()
            f.close()

        with open(os.path.join(path, 'gromacs.top'), 'w+') as fp:
            for eachline in alllines:
                s = re.sub('CH4                 ', 'CH4                     %s'%All[0],eachline)
                fp.writelines(s)

        ################################################################################delete_box
        x = [j for j in range(All[0]+2, All[3] + 2)]
        print(x)
        for i in x:
            with open(os.path.join(path, 'box.g96'), 'r') as f:
                lines = f.readlines()

            with open(os.path.join(path, 'box.g96'), 'w') as f:
                for l in lines:
                    if ' %s CH4' % i not in l:
                        f.write(l)

    rootpath = os.path.dirname(sys.path[0])
    os.chdir(rootpath)