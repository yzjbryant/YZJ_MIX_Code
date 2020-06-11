#include <stdio.h>
int main(){
    int max;
    printf("Enter: ");
    scanf("%d",&max);


    int number=max;
    while (
           number>0&&
           !(number%3==2&&number%5==3&&number%7==2)){
        --number;
    }
    if (number>0){
        printf("%d\n",number);
    }
    return 0;
}

#include <stdio.h>
int main(){
    int max;
    printf("Enter: ");
    scanf("%d",&max);

    int answer=0;
    int number;
    for (number=max;number>=1&&answer==0;--number){
        if (number%3==2&&number%5==3&&number%7==2){
            answer=number;
        }
    }
    if (answer!=0){
        printf("%d\n",answer);
    }
    return 0;
}


