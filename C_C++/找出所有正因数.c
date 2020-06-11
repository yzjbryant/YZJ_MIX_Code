#include <stdio.h>
int main(){
    int number;
    int N;
    printf("Please enter: ");
    scanf("%d",&N);
    for (number=1;number<=N;number++){
        if (N%number==0){
            printf("%d\n",number);
        }
    }
    printf("\n");
    return 0;
}
