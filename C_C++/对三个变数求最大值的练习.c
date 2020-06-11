#include <stdio.h>
int main(){
    int a,b,c,max;
    printf("Enter: ");
    scanf("%d%d%d",&a,&b,&c);
    printf("Max is %d\n",max3(a,b,c));
    return 0;
}

int max3(int a,int b,int c){
    return max2(max2(a,b),c);
}

int max2(int a,int b){
    if (a>=b){
        return a;
    }else{
        return b;
    }
}

