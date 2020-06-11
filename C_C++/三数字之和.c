#include <stdio.h>

int main(){
    int integer,sum;
    printf("Please enter first integer: ");
    scanf("%d",&integer);
    sum=integer;
    printf("Please enter second integer: ");
    scanf("%d",&integer);
    sum=sum+integer;
    printf("Please enter third integer: ");
    scanf("%d",&integer);
    sum=sum+integer;
    printf("Sum is %d.\n",sum);
    return 0;
}
