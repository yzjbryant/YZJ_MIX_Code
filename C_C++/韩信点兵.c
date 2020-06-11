#include <stdio.h>
int main(){
    int number=1;
    while (!(number%3==2&&number%5==3&&number%7==2)){
        ++number;
    }
    printf("%d\n",number);
    return 0;
}
