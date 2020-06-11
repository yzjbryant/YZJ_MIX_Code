Number=[1392]
Last_Number=[4]

Update_Last_Number=[]

New_last_number=[]

if (Number[0]-Last_Number[0])%3 == 0:
    x=Number[0]-(Number[0]-Last_Number[0])/3
    y=Number[0]-(Number[0]-Last_Number[0])/3*2
    print(Number[0],x,y,Last_Number[0])
else:
    z=(Number[0]-Last_Number[0])%3
    Update_Last_Number.append(z)
    u=Last_Number[0]+Update_Last_Number[0]
    New_last_number.append(u)
    if (Number[0]-New_last_number[0])%3 == 0:
        x = Number[0] - (Number[0] - New_last_number[0]) / 3
        y = Number[0] - (Number[0] - New_last_number[0]) / 3 * 2
        print(Number[0],x, y, New_last_number[0])

