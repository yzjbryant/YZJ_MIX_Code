def bubble(bubblelist):
    listLength=len(bubblelist)
    while listLength>0:
        for i in range(listLength-1):
            if bubblelist[i]>bubblelist[i+1]:
                tmp=bubblelist[i]
                bubblelist[i]=bubblelist[i+1]
                bubblelist[i+1]=tmp
        listLength-=1
    print(bubblelist)
if __name__=='__main__':
    I=[1,2,5,1,4,2,5,7,0]
    bubble(I)