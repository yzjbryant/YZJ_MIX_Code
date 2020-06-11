#include <stdio.h>
int main(){
    int N;
    printf("N= ");
    scanf("%d",&N);

    int isPrime=1;
    int number;
    for (number=2;number<N&&isPrime;++number){
        if (N%number==0){
            isPrime=0;
        }
    }
    if (isPrime==1){
        printf("Yes\n");
    }else{
        printf("No\n");
    }


    return 0;
}

