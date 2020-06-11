#include <stdio.h>
int main(){
    int answer=4;
    int guess;

    do {
        printf("Please enter: ");
        scanf("%d",&guess);
        if (guess>answer){
            printf("Too Large!\n");
        }else if (guess<answer){
            printf("Too Small!\n");
        }else{
            printf("Correct!\n");
        }
    }while (guess!=answer);
    return 0;
}

