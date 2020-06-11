import os

path='F:\Ds_analyze'
with open(os.path.join(path,'genbox.sh'),'w') as f:
    str1 = "#/bin/bash\n"
    f.write(str1)
    for i in range(137500,137838):
        str2='qdel %s\n'%i
        f.write(str2)