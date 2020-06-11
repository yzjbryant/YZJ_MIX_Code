#include <stdio.h>
void f(int[3]);


int main(){
    int v[3]={1,2,3};
    f(v);
    return 0;
}
void f(int v[3]){
    printf("max is %zu\n",sizeof(int));
    printf("max is %zu\n",sizeof(v[0]));
    printf("max is %zu\n",sizeof(v)/sizeof(v[0]));
}
