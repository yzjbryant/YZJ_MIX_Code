#include <stdio.h>
int main(){
    int answer=4;
    int guess;
    printf("Please input one: ");
    scanf("%d",&guess);
    if (guess>answer){
        printf("1\n");
    }
    if (guess<answer){
        printf("2\n");
    }
    if (guess==answer){
        printf("3\n");
    }
    return 0;
}
