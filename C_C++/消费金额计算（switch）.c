#include <stdio.h>
int main(){
    int total=0;
    int id;
    do {
        scanf("%d",&id);
        switch (id){
            case 1:total += 90;break;
            case 2:total += 75;break;
            case 3:total += 83;break;
            case 4:total += 89;break;
            case 5:total += 71;break;
        }
    } while (id!=0);
    printf("Total is %d\n",total);

    return 0;
}
