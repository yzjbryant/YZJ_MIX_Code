#include <stdio.h>
int main(){
    int N;
    printf("Please enter: ");
    scanf("%d",&N);

    int i,j;
    for (i=1;i<=N;++i){
        for (j=1;j<=N;++j){
            printf("*");
        }
    }
    return 0;
}
