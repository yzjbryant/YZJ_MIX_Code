import phone
import pandas as pd

phone_list=[]
for i in range(1300000,1300010):
    try:
        info=phone.Phone().find(str(i))
        phone_list.append(info.values())
        #如果出错，推出本次循环
        continue
    except(RuntimeError,TypeError,NameError,AttributeError):
        #出错后，直接跳过，不做任何处理
        pass

df=pd.DataFrame(phone_list,columns=['手机号段','省份','运营商','邮编','区号','地市'])
print(df)

df.to_csv('phone_list.csv',encoding='gbk')
