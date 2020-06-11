#include <stdio.h>

int main(){
    int integer1;
    int integer2;
    int sum;
    printf("Please enter the first integer: ");
    scanf("%d",&integer1);
    printf("Please enter the second integer: ");
    scanf("%d",&integer2);
    sum=integer1+integer2;
    printf("Sum is %d.\n",sum);
    return 0;
}
