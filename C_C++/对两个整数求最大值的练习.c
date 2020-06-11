#include <stdio.h>
int main(){
    int a,b,max;
    printf("a= ");
    scanf("%d",&a);
    printf("b= ");
    scanf("%d",&b);
    printf("The max is %d.\n",max2(a,b));
    return 0;
}

int max2(int a, int b){
    int max;
    if (a>=b){
        max=a;
    }else {
        max=b;
    }
    return max;
}
