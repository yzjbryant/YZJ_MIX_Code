#include <stdio.h>
int main(){
    int a,b,max;
    printf("Please enter a: ");
    scanf("%d",&a);

    printf("Please enter b: ");
    scanf("%d",&b);
    max=a;
    if (max<b){
        max=b;
    }

    printf("The max is %d\n",max);
    return 0;

}
