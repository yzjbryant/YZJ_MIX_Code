#include <stdio.h>

int main(){
    int integer1,integer2,integer3;
    printf("Please enter integer1: ");
    scanf("%d",&integer1);
    printf("Please enter integer2: ");
    scanf("%d",&integer2);
    printf("Please enter integer3: ");
    scanf("%d",&integer3);
    double average = (integer1 + integer2 + integer3) / 3.;
    printf("average is %f\n",average);
    return 0;
}
