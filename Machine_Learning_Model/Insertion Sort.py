
def insert_sort(insertlist):
    for i in range(len(insertlist)):
        min_index=i
        for j in range(i+1,len(insertlist)):
            if insertlist[min_index]>insertlist[j]:
                min_index=j
            tmp=insertlist[i]
            insertlist[i]=insertlist[min_index]
            insertlist[min_index]=tmp
        #     print(str(insertlist))
        # print("result: "+str(insertlist))

if __name__=='__main__':
    I=[1,2,3,4,55,6,1,0]
    insert_sort(I)
    print(I)