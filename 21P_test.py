import torch
import torch.nn as nn
import numpy as np
import random
import gym

# 创建 DQNAgent 类
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        # 创建一个具有两个隐藏层的神经网络模型
        model = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size)
        )
        return model

    def act(self, state):
        for i in state:
            if type(i)==dict:
                state=state[:-1]
        state_array = np.array([int(x) if isinstance(x, str) else x for x in state])
        state_array = state_array.astype(float)
        state_tensor = torch.from_numpy(state_array).float()
        q_values = self.model(state_tensor)
        return q_values.argmax().item()
def load_model(path):
    # 加载已经训练好的模型
    model = DQNAgent(state_size=3, action_size=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
def predict(model,state):
    # 将数据转换为张量，并进行预测
    with torch.no_grad():
        prediction = model.act(state)

    if prediction==0:
        print("不要手牌")
    elif prediction==1:
        print("再拿一张手牌")
    else:
        print("其他")

        
# 假设你有一些待预测的数据 state    (自己的点数，庄家的点数，用不用Soft计算)
"""
如果使用 Ace 的 Soft 计算后，手牌点数不超过 21 分，则该 Ace 会被视作 11 分。
如果使用 Ace 的 Soft 计算后，手牌点数超过 21 分，则该 Ace 会被视作 1 分。
如果一个手牌中有多个 Ace，那么只能有一个 Ace 被视作 11 分，其他 Ace 都被视作 1 分。

"""
predict(load_model('dqn_21.pt'),((17, 19, False)))


