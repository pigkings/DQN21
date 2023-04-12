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
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Linear(64,16),
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
    model = DQNAgent(state_size=156, action_size=2)
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

def _get_pokers_now(deck1):
    deck = [(rank, suit) for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] for suit in ["spades", "hearts", "diamonds", "clubs"]]
    state = np.zeros(4*13)
    for i in deck1:
        state[deck.index((i[0],i[1]))]+=1
    return state

def _get_pokers_vl(rank, suit):
    deck = [(rank, suit) for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] for suit in ["spades", "hearts", "diamonds", "clubs"]]
    return deck.index((rank, suit))
def get_state(num_decks,player_hand,dealer_hand,history):
    deck=num_decks * [(rank, suit) for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] for suit in ["spades", "hearts", "diamonds", "clubs"]]
    deck1 = [card for card in deck if card not in player_hand]
    deck1 = [card for card in deck1 if card not in dealer_hand]
    deck1 = [card for card in deck1 if card not in history]
    state = np.zeros((4*13, 2))
    now_state=_get_pokers_now(deck1)
    for i in range(len(player_hand)):
        rank, suit = player_hand[i]
        if i == 0:
            state[_get_pokers_vl(rank, suit)][0] += 1
            state[_get_pokers_vl(dealer_hand[0][0],dealer_hand[0][1])][0] -=1
        else:
            state[_get_pokers_vl(rank, suit)][1] += 1
    state=np.insert(state, 2, now_state, axis=1)
    state=state.flatten()
    state=np.round(state).astype(int).tolist()
    state = tuple(state)
    return state

num_decks=1
player_hand=[("Q","spades"),("K","spades")]
dealer_hand=[("J","spades")]
history=[("6","spades"),("9","spades"),('2', 'spades')]
predict(load_model('model/dqn_21_9000.pt'),get_state(num_decks,player_hand,dealer_hand,history))


