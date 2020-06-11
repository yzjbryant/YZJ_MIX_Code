#include <stdio.h>
int main(){
    int a,b,max;
    printf("enter: ");
    scanf("%d%d",&a,&b);
    if (a>=b){
        max=a;
    }else {
        max=b;
    }
    printf("Max is %d.\n",max);
    return 0;
}
