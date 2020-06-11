#include <stdio.h>
int main(){
    int  num;
    int  sum=0;

    printf("Please enter: ");
    scanf("%d",&num);

    while (num!=0){
        sum=sum+num;
        scanf("%d",&num);
    }
    printf("The sum is %d\n",sum);
    return 0;
}
