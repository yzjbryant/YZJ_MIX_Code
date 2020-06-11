#include <stdio.h>
int main(){
    int i,j;
    for (i=2;i<=9;++i){
        for (j=1;j<=9;++j){
            printf("%d*%d=%d\n",i,j,i*j);
        }
    }
    return 0;
}

#include <stdio.h>
int main(){
    int c;
    for (c=1;c<=72;++c){
        int i=(c-1)/9+2;
        int j=(c-1)%9+1;
        printf("%d*%d=%d\n",i,j,i*j);
    }
    return 0;
}
