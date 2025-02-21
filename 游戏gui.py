import time
import gym
import pygame
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

# 初始化环境
pygame.init()
环境 = gym.make("CartPole-v1", render_mode="human")

关键参数, _ = 环境.reset()

小车位置 = 关键参数[0]
小车速度 = 关键参数[1]
杆角度 = 关键参数[2]
杆尖端速度 = 关键参数[3]

print(小车位置)
print(小车速度)
print(杆角度)
print(杆尖端速度)

游戏开始时间 = time.time()
最大游戏次数 = 1000

步数 = 0
fail = False


# 定义策略网络（全连接神经网络）
class 策略网络(nn.Module):
    def __init__(self):
        super(策略网络, self).__init__()
        self.全连接1 = nn.Linear(4, 128)
        self.全连接2 = nn.Linear(128, 2)  # 输出两个动作的概率
        self.激活函数 = nn.ReLU()
        self.输出层 = nn.Softmax(dim=-1)  # 归一化为概率分布

    def forward(self, 状态):
        隐藏层 = self.激活函数(self.全连接1(状态))
        动作概率 = self.输出层(self.全连接2(隐藏层))
        return 动作概率




# 创建策略网络和优化器
策略 = 策略网络()
优化器 = optim.Adam(策略.parameters(), lr=0.01)
状态 = 环境.reset()[0]
for 步数 in range(1, 最大游戏次数 + 1):

    time.sleep(0.1)

    状态张量 = torch.tensor(状态, dtype=torch.float32)
    动作概率 = 策略(状态张量)
    行动参数 = torch.argmax(动作概率).item()  # 按概率采样动作

    新状态, _, done, _, _ = 环境.step(行动参数)
    状态 = 新状态
    print(行动参数)
    print(步数)
    print(done)

    if done:
        fail = True
        break
