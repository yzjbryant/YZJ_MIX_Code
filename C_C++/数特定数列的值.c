#include <stdio.h>
int main(){
    int count;
    for (count=2; count<=10; count+=2){
        int  number = 2*count;
        if (number%3!=0){
            printf("%d\n",number);
        }
    }
    return 0;
}


