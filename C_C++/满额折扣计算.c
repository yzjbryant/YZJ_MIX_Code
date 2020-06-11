#include <stdio.h>
int main(){
    int p,m,original;
    printf("Please enter people: ");
    scanf("%d",&p);
    original=300*p;
    if (original<3000){
        money=original;
        printf("qian is %d\n",money);
    }
    if (original>=3000){
        money=original*0.8;
        printf("qian is %d\n",money);
    }
    return 0;
}
