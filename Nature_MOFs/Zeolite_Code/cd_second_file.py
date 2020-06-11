import os

import sys
def GetVideoName(dir):
    listName = []
    for fileName in os.listdir(dir):
        if os.path.splitext(fileName)[1] == '.cif':
            fileName = os.path.splitext(fileName)[0]
            listName.append(fileName)
    return listName

file_list = GetVideoName("./CIF")
print(file_list)
for j in file_list:
    os.chdir("%s" %j)
    os.system("mkdir 1 2 3 4")
    # rootpath = os.path.dirname(sys.path[0])
    # os.chdir(rootpath)