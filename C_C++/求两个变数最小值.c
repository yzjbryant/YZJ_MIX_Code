#include <stdio.h>
int main(){
    int a,b,min;
    printf("Please enter number: ");
    scanf("%d%d",&a,&b);
    min=a;
    if (b<min){
        min=b;
    }
    printf("The min is %d.\n",min);
    return 0;
}
