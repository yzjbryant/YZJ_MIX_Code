#include <stdio.h>

int main(){
    char ch1,ch2;
    scanf("%c",&ch1);
    ch2 = ch1 + ('a'-'A');
    printf("output: %c\n",ch2);
    return 0;
}
