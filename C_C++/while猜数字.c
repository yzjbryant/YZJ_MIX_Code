#include <stdio.h>
int main(){
    int answer=4;
    int guess;
    printf("Please enter: ");
    scanf("%d",&guess);
    if (guess>answer){
            printf("Too large!\n");
        }else if (guess<answer){
            printf("Too small!\n");
        }else {
            printf("Correct!");
        }
    while (guess!=answer){
        printf("Please enter: ");
        scanf("%d",&guess);
        if (guess>answer){
            printf("Too large!\n");
        }else if (guess<answer){
            printf("Too small!\n");
        }else {
            printf("Correct!");
        }
    }
    return 0;
}
