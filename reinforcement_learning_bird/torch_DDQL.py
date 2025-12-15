#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flappy Bird Double DQN - PyTorch 版本

依赖建议：
- python == 3.10
- torch  >= 2.0 (你可以用 2.2.1；如果 3.10 上装不到 2.3.0 就用 2.2.1)
- numpy == 1.26.4
- opencv-python == 4.9.0 或 4.8.x
- pygame == 2.5.2

环境：
- 使用原来的 game/wrapped_flappy_bird.py 里的 GameState()
"""

from __future__ import print_function
import os
import sys
import cv2
import time
import math
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append("game/")
from wrapped_flappy_bird import GameState

# ==================== 超参数 ====================
GAME = "bird"
ACTIONS = 2                 # 动作数：不跳 / 跳
GAMMA = 0.99               # 折扣因子
OBSERVE = 1000             # 纯观察步数（只收集经验，不训练）
EXPLORE = 300000           # epsilon 从 INITIAL 衰减到 FINAL 所花的步数
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 100000     # 经验池容量
BATCH = 32
FRAME_PER_ACTION = 4
LEARNING_RATE = 2.5e-4
TARGET_UPDATE_FREQ = 2000  # target 网络同步频率（步数）
SAVE_INTERVAL = 20000      # 保存模型的步数间隔

# ==================== 设备选择 ====================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("Using device:", device)


# ==================== 网络结构 ====================
class DQN(nn.Module):
    """
    CNN 结构尽量贴近你原来的 TF 版：
    - Conv1: 4 → 32, kernel=8, stride=4, padding=2
    - MaxPool: 2x2, stride=2
    - Conv2: 32 → 64, kernel=4, stride=2, padding=1
    - Conv3: 64 → 64, kernel=3, stride=1, padding=1
    - Flatten 后尺寸为 64 * 5 * 5 = 1600
    - FC1: 1600 -> 512
    - FC2: 512 -> ACTIONS
    """

    def __init__(self, num_actions=ACTIONS):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 动态算 flatten size，避免硬编码错误
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 80, 80)
            x = self.conv1(dummy)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flatten_size = x.view(1, -1).size(1)
        # 一般应该是 1600
        print("Flatten size =", self.flatten_size)

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: [B, 4, 80, 80], 像素 0~255
        x = x / 255.0  # 归一化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)  # [B, ACTIONS]
        return q_values


# ==================== Replay Memory ====================
class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state / next_state: np.array, 形状 [4, 80, 80]
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==================== 图像预处理 ====================
def preprocess(frame):
    """
    输入：RGB frame (H, W, 3)
    输出：灰度二值化 (80, 80), float32
    """
    frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
    return frame.astype(np.float32)


# ==================== 训练主循环 ====================
def train():
    env = GameState()

    # 主网络 & 目标网络
    main_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAY_MEMORY)

    print("🚀 Double DQN training started!")

    # 初始状态：do nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    frame, _, terminal = env.frame_step(do_nothing)

    frame = preprocess(frame)
    # 初始 state: 4 帧相同
    state = np.stack([frame] * 4, axis=0)  # [4,80,80]

    epsilon = INITIAL_EPSILON
    t = 0  # 全局步数

    while True:
        # ========== 1. 选择动作（ε-greedy） ==========
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # [1,4,80,80]
        q_values = main_net(state_tensor)
        q_values_np = q_values.detach().cpu().numpy()[0]

        action_index = 0
        action_onehot = np.zeros(ACTIONS)

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                # 探索：减少随机跳的比例
                jump_random_prob = 0.20
                if random.random() < jump_random_prob:
                    action_index = 1  # 跳
                else:
                    action_index = 0  # 不跳
            else:
                # 利用：选 Q 最大的动作
                action_index = int(np.argmax(q_values_np))
        else:
            # 非动作帧：默认不跳
            action_index = 0

        action_onehot[action_index] = 1

        # epsilon 线性退火
        if t > OBSERVE and epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / float(EXPLORE)
            epsilon = max(FINAL_EPSILON, epsilon)

        # ========== 2. 执行动作，得到新帧和 reward ==========
        next_frame_color, r, done = env.frame_step(action_onehot)

        # 奖励整形
        if done:
            r = -1.0
        elif r == 1:
            r = 1.0
        else:
            r = 0.0002

        next_frame = preprocess(next_frame_color)
        next_state = np.concatenate(
            ([next_frame], state[:-1]), axis=0
        )  # 新帧在最前 [4,80,80]

        # ========== 3. 存入经验池 ==========
        memory.push(state, action_index, r, next_state, done)
        state = next_state
        t += 1

        # ========== 4. 从经验池采样并训练（Double DQN 核心） ==========
        if t > OBSERVE and len(memory) >= BATCH:
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
            ) = memory.sample(BATCH)

            # 转成 tensor
            batch_states_t = torch.from_numpy(batch_states).to(device)          # [B,4,80,80]
            batch_actions_t = torch.from_numpy(batch_actions).to(device)        # [B]
            batch_rewards_t = torch.from_numpy(batch_rewards).to(device)        # [B]
            batch_next_states_t = torch.from_numpy(batch_next_states).to(device)# [B,4,80,80]
            batch_dones_t = torch.from_numpy(batch_dones).to(device)            # [B]

            # --- Q(s,a) from main_net ---
            q_values = main_net(batch_states_t)                 # [B,2]
            # 选出对应动作的 Q(s,a)
            q_selected = q_values.gather(1, batch_actions_t.unsqueeze(1)).squeeze(1)

            # --- Double DQN: 主网选 a_max，目标网评估 ---
            # 主网络在 s' 上选动作
            q_next_main = main_net(batch_next_states_t)         # [B,2]
            next_actions = q_next_main.argmax(dim=1)            # [B]

            # 目标网络在 s' 上评估这些动作
            q_next_target = target_net(batch_next_states_t)     # [B,2]
            q_next_selected = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # 对终止状态，未来回报为 0
            targets = batch_rewards_t + GAMMA * q_next_selected * (1.0 - batch_dones_t)

            # Huber loss（Smooth L1）
            loss = F.smooth_l1_loss(q_selected, targets.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ========== 5. 更新目标网络 ==========
        if t % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(main_net.state_dict())
            print("Target network updated at step", t)

        # ========== 6. 保存模型 ==========
        if t % SAVE_INTERVAL == 0:
            os.makedirs("saved_networks", exist_ok=True)
            save_path = os.path.join("saved_networks", f"{GAME}-double-dqn-{t}.pth")
            torch.save(main_net.state_dict(), save_path)
            print("💾 模型已保存:", save_path, "| 当前 ε = {:.3f}".format(epsilon))

        # ========== 7. 打印训练状态 ==========
        if t <= OBSERVE:
            phase = "observe"
        elif t <= OBSERVE + EXPLORE:
            phase = "explore"
        else:
            phase = "train"

        print(
            "Step {} | {} | ε = {:.3f} | Action = {} | Reward = {:.4f}".format(
                t, phase, epsilon, action_index, r
            )
        )


if __name__ == "__main__":
    train()