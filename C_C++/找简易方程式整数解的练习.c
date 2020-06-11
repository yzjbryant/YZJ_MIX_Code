#include <stdio.h>
int main(){


    int i,j;
    for (i=1;i<=30;++i){
        for (j=1;j<=30;++j){
            if (i+j==30&&i*j==221&&i<=j){
                printf("%d,%d\n",i,j);
            }

        }
    }
    return 0;
}



