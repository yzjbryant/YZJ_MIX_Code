#include <stdio.h>
int main(){
    int max;
    printf("Enter :");
    scanf("%d",&max);

    int number;
    for (number=max;number>=1&&answer==0;--number){
        if (number%3==2&&number%5==3&&number%7==2){
            break;
        }
    }
    if (answer!=0){
        printf("%d\n",answer);
    }


    return 0;
}


