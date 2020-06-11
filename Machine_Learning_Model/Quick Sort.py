def quickSort(L,low,high):
    i=low
    j=high
    if i>=j:
        return L
    key=L[i]
    while i<j:
        while i<j and L[j]>=key:
            j=j-1
        L[i]=L[j]
        while i<j and L[i]<=key:
            i=i+1
        L[j]=L[i]
    L[i]=key
    quickSort(L,low,i-1)
    quickSort(L,j+1,high)
    print(L)
    return L
if __name__=='__main__':
    I=[1,2,34,5,6,7]
    quickSort(I,1,5)