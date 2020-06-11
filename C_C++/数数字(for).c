#include <stdio.h>
int main(){
    int M,N;
    printf("Please enter M: ");
    scanf("%d\n",&M);
    printf("Please enter N: ");
    scanf("%d\n",&N);
    int count;
    for (count=M; count<=N; count++){
        printf("%d\n",count);
    }
    return 0;
}
