#include <stdio.h>
int main(){
    int answer=4;
    int guess;
    printf("Please enter: ");
    scanf("%d",&guess);

    while (guess!=answer){

        if (guess>answer){
            printf("Too large!\n");
        }else {
            printf("Too small!\n");
        }
        printf("Please enter: ");
        scanf("%d",&guess);
    }
    printf("Correct\n");
    return 0;
}

