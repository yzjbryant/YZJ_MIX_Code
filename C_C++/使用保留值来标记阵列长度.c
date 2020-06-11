#include <stdio.h>
int main(){
    int  v[5]={1,2,3,-1,6};
    printf("%d\n",length(v));
    return 0;
}
int length(int v[]){
    int i=0;
    while (v[i] != 6){
        i++;
    }
    return i+1;
}
