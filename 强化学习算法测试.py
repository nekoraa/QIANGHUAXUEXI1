import time

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 创建环境
环境 = gym.make("CartPole-v1")


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
优化器 = optim.Adam(策略.parameters(), lr=0.001)


# 计算折扣奖励
def 计算折扣奖励(奖励序列, 折扣因子=0.99):
    折扣奖励 = []
    累计奖励 = 0
    for 奖励 in reversed(奖励序列):
        累计奖励 = 奖励 + 折扣因子 * 累计奖励
        折扣奖励.insert(0, 累计奖励)  # 逆序存储
    折扣奖励 = torch.tensor(折扣奖励)
    折扣奖励 = (折扣奖励 - 折扣奖励.mean()) / (折扣奖励.std() + 1e-9)  # 归一化
    return 折扣奖励


# 训练策略网络
训练轮数 = 100000
for 轮次 in range(训练轮数):
    状态 = 环境.reset()[0]
    轨迹 = []

    # 采样一条轨迹
    for _ in range(10000):
        状态张量 = torch.tensor(状态, dtype=torch.float32)
        动作概率 = 策略(状态张量)
        动作 = torch.multinomial(动作概率, 1).item()  # 按概率采样动作

        新状态, 奖励z, 结束, _, _ = 环境.step(动作)
        奖励 = 10
        轨迹.append((状态, 动作, 奖励))
        状态 = 新状态

        if 结束:
            break

    # 计算折扣奖励
    状态批次, 动作批次, 奖励批次 = zip(*轨迹)
    折扣奖励批次 = 计算折扣奖励(奖励批次)

    # 计算策略梯度损失
    总损失 = 0
    for i in range(len(轨迹)):
        状态张量 = torch.tensor(状态批次[i], dtype=torch.float32)
        动作张量 = torch.tensor(动作批次[i])
        奖励张量 = 折扣奖励批次[i]

        动作概率 = 策略(状态张量)
        选中动作概率 = 动作概率[动作张量]
        损失 = -torch.log(选中动作概率) * 奖励张量  # 策略梯度公式
        总损失 += 损失

    # 反向传播优化
    优化器.zero_grad()
    总损失.backward()
    优化器.step()

    # 打印训练信息
    if 轮次 % 10 == 0:
        print(f"轮次: {轮次}, 轨迹长度: {len(轨迹)}")
    if len(轨迹) == 10000:
        break



测试环境 = gym.make("CartPole-v1", render_mode="human")
# 测试智能体
状态 = 测试环境.reset()[0]
总奖励 = 0
while True:
    time.sleep(0.01)
    测试环境.render()
    状态张量 = torch.tensor(状态, dtype=torch.float32)
    动作概率 = 策略(状态张量)
    动作 = torch.argmax(动作概率).item()  # 选择最大概率的动作
    状态, 奖励, 结束, _, _ = 测试环境.step(动作)
    总奖励 += 奖励
    if 结束:
        break

print(f"测试完成，总奖励: {总奖励}")
环境.close()
测试环境.close()