#include <stdio.h>
int maxv(int[],int N);

int main(){
    int a[3]={3,6,9};
    printf("Max: %d\n",maxv(a,3));
    int b[5]={1,2,3,6,9};
    printf("Max: %d\n",maxv(b,5));
    return 0;
}

int maxv(int v[],int N){
    int max=v[0],i;
    for (i=1;i<N;i++){
        if (v[i]>max){
            max=v[i];
        }
    }
    return max;
}
