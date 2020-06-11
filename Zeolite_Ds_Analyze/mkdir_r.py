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
for i in range(len(file_list)):
    os.system("mkdir %s "%file_list[i])

