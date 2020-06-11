#include <stdio.h>
int main(){
    int min,max;
    printf("Enter: ");
    scanf("%d",&min);
    printf("Enter: ");
    scanf("%d",&max);

    int number;
    for (number=max;number>=min;number--){
        if (number%3==2&&number%5==3&&number%7==2){
            printf("%d ",number);
        }
    }
    printf("\n");
    return 0;
}

