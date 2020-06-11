import phone
import pandas as pd

phone_list=[]
for i in range(1585100,1585109):
    try:
        info=phone.Phone().find(str(i))
        phone_list.append(info.values())
        continue
    except(RuntimeError,NameError,AttributeError,TypeError):
        pass
df=pd.DataFrame(phone_list,columns=['1','2','3','4','5','6'])
df.to_csv('1.csv',encoding='gbk')