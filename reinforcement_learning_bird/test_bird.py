#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("game/")
from wrapped_flappy_bird import GameState


# ---------------- 网络结构（与训练时保持一致） ----------------
class DQN(nn.Module):
    def __init__(self, num_actions=2):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 4, 80, 80)
            x = self.pool1(self.conv1(dummy))
            x = self.conv2(x)
            x = self.conv3(x)
            self.flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q


# ---------------- 图像预处理 ----------------
def preprocess(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
    return frame.astype(np.float32)


# =====================================================
#                   测试主逻辑
# =====================================================
def test(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Using device:", device)

    # 初始化网络
    net = DQN().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    print("✅ 模型已加载:", model_path)

    # 初始化游戏
    env = GameState()

    # do nothing
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    frame, _, terminal = env.frame_step(do_nothing)

    frame = preprocess(frame)
    state = np.stack([frame] * 4, axis=0)

    while True:
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = net(state_tensor)
        action_index = int(torch.argmax(q_values[0]).item())

        action = np.zeros(2)
        action[action_index] = 1

        # 执行动作
        next_frame, reward, terminal = env.frame_step(action)

        next_frame = preprocess(next_frame)
        next_state = np.concatenate(([next_frame], state[:-1]), axis=0)
        state = next_state

        # 如果死了，重新开始
        if terminal:
            print("💀 Bird died, restarting...")
            frame, _, terminal = env.frame_step(do_nothing)
            frame = preprocess(frame)
            state = np.stack([frame] * 4, axis=0)


if __name__ == "__main__":
    test("saved_networks/bird-double-dqn-1100000.pth")