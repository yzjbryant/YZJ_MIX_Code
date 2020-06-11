#include <stdio.h>
int main(){
    int a,b,c,d,max;
    printf("Please enter number: ");
    scanf("%d%d%d%d",&a,&b,&c,&d);
    max=a;
    if (b>a&&b>=c&&b>=d){
        max=b;
    }
    if (c>a&&c>b&&a>=d){
        max=c;
    }
    if (d>a&&d>b&&d>c){
        max=d;
    }
    printf("The max is %d.\n",max);
    return 0;
}
