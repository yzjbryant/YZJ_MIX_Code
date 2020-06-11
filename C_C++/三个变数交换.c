#include <stdio.h>
int main(){
    int a,b,c,t;
    scanf("%d %d %d\n",&a,&b,&c);
    printf("Before: %d %d %d\n",&a,&b,&c);
    if (a<c&&c<b){
        t=b;b=c;c=t;
    }
    if (b<a&&a<c){
        t=a;a=b;b=c;c=t;
    }
    if (c<a&&a<b){
        t=a;a=c;c=b;b=t;
    }
    if (c<b&&b<a){
        t=a;a=c;c=t;
    }
    printf("After: %d %d %d",a,b,c);
    return 0;
}
