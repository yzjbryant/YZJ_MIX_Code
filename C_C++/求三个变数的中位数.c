#include <stdio.h>
int main(){
    int a,b,c,mid;
    printf("Please enter number: ");
    scanf("%d%d%d",&a,&b,&c);
    mid=a;
    if (a<=b&&b<=c||c<=b&&b<=a){
        mid=b;
    }
    if (a<=c&&c<=b||b<=c&&c<=a){
        mid=c;
    }


    printf("The mid is %d.\n",mid);
    return 0;
}
