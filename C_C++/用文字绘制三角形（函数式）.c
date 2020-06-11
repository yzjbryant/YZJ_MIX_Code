#include <stdio.h>
void print_stars(int i);
void print_triangle(int N);

int main(){
    int N;
    printf("Enter: ");
    scanf("%d",&N);
    print_triangle(N);
    return 0;
}

void print_triangle(int N){
    int i;
    for (i=1;i<=N;++i){
        print_stars(i);
        printf("\n");
    }
}

void print_stars(int i){
    int j;
    for (j=1;j<=i;++j){
        printf("*");
    }
}

