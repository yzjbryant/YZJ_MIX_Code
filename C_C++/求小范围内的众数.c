#include <stdio.h>
int main(){
    int i,n,b[10]={0};
    for (i=1;i<=10;i++){
        scanf("%d",&n);
        b[n]++;
    }
    int ans=0;
    for (n=0;n<10;n++){
        if (b[n]>=b[ans]){
            ans=n;
        }
    }
    printf("Ans is %d\n",ans);
    return 0;
}
