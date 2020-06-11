/*
1.数据设计
2.模块化设计
3.实现过程
*/
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>     //光标设置的API
#include <time.h>        //食物随机
#include <conio.h>       //按键监控
/////////////////////////////
//       辅助宏定义
#define MAPHEIGHT 25     //窗口的属性
#define MAPWIDTH  60
#define SNAKESIZE 50     //蛇的最大节数




//数据设计
/////////////////////////////
//       食物的结构体
struct
{
    //如何去定位：坐标
    int x;
    int y;
}food;
/////////////////////////////
//       蛇的结构体
struct
{
    //记录每一节蛇的坐标
    int x[SNAKESIZE];
    int y[SNAKESIZE];
    int len;         //蛇的长度
    int speed;       //蛇的移动速度
}snake;

/////////////////////////////
//       全局变量
int key='w';     //初始化移动方向
int changeFlag=0;//蛇的变化的标记


//模块化设计---》功能的划分---》抽象到函数
/////////////////////////////////////////////
//               怎么抽象：具体了解业务逻辑
/////////////////////////////////////////////
//1.画地图
void drawMap()
{   //♥：食物     🐀𕐀：蛇身
    srand((unsigned int)time(NULL));  //随机函数种子

    //1.圈地
    //1.1左右边框
    for (int i=0;i<=MAPHEIGHT;i++){
        gotoxy(0,i);
        printf("🐀𕐀")；
        gotoxy(MAPHEIGHT, i);
        printf("🐀𕐀"；)
    }
    //1.2上下边框
    //因为🐀𕐀z占用两个字符
    for (int i=0;i<=MAPWIDTH;i+=2){
        gotoxy(i,0);
        printf("🐀𕐀")；
        gotoxy(i,MAPWIDTH);
        printf("🐀𕐀"；)
    //2.画蛇
    //2.1确定蛇的属性
    snake.len=3;
    snake.speed=100;
    snake.x[0]=MAPWIDTH/2;
    snake.y[0]=MAPHEIGHT/2;
    //3.画食物
        //3.1确定坐标
    food.x=rand()%(MAPWIDTH-4)+2;
    food.y=rand()%(MAPHEIGHT-2)+1;
        //3.2画出来就可以
    gotoxy(food.x,food.y);
    printf("❤")
}
//2.食物的产生
void createFood()
{

}
//3.按键操作
void keyDown()
{

}
//4.蛇的状态：判断是否结束游戏
int snakeStatus()
{
    return 1;
}
//5.辅助函数：光标移动
void gotoxy(int x,int y);   //TC 是有的，现在已经淘汰了，要自己实现
{
    //调用win32 API去设置控制台的光标位置
    //1.找到控制台的这个窗口
    HANDLE handle=GetStdHandle(STD_OUTPUT_HANDLE);
    //2.光标的结构体
    COORD coord;
    //3.设置坐标
    coord.X=x;
    coord.Y=y;
    //4.同步到控制台Set Console Cursor Position
    SetConsoleCursorPosition(handle,coord);
}









int main()
{
    gotoxy(20,20)





    printf("GameOver\t");
    system("pause");
    return 0;
}
