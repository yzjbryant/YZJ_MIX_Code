import pygame

# 游戏的初始化
pygame.init()

# 创建游戏的窗口 480*700
screen = pygame.display.set_mode((480,700))

# 绘制背景图像
background = pygame.image.load("./images/background.png")
screen.blit(background,(0,0))
# pygame.display.update()

# 绘制英雄的飞机
hero = pygame.image.load("./images/me1.png")
screen.blit(hero,(200,500))

#可以在所有绘制工作完成之后，统一调用update方法
pygame.display.update()

#游戏循环-> 意味着游戏的正式开始

while True:#无限循环

    pass

pygame.quit()

