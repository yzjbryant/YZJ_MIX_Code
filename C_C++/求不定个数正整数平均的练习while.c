#include <stdio.h>
int main(){
    int num;
    int sum=0;
    int count=0;
    float average;
    printf("Please enter:  ");
    scanf("%d",&num);

    while  (num!=0){
        sum=sum+num;
        count+=1;
        scanf("%d",&num);


    }
    average=(float) sum/count;
    printf("The average is %f\n",average);
    return 0;
}
