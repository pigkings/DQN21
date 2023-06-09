import random
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BlackjackEnv:
    def __init__(self, num_decks):
        self.num_decks = num_decks
        self.suits = ["spades", "hearts", "diamonds", "clubs"]
        self.ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.deck = self.num_decks * [(rank, suit) for rank in self.ranks for suit in self.suits]
        self.player_hand = []#玩家手牌
        self.dealer_hand = []
        self.playernum=2
        
    def reset(self):
        #重新洗牌发牌
        if len(self.deck)<=self.playernum*9:
            self.deck = self.num_decks * [(rank, suit) for rank in self.ranks for suit in self.suits]
        self.player_hand = []
        self.dealer_hand = []
        self._deal_cards()#发牌
        return self._get_state()

    def step(self, action):
        if action == 0:  # hit拿牌
            self.player_hand.append(self.deck.pop())
            if self._is_bust(self.player_hand):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        elif action == 1:  # stand
            done = True
            while self._get_hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.deck.pop())
            if self._is_bust(self.dealer_hand):
                reward = -1#牌的点数大过21点，牌爆掉了
            elif self._get_hand_value(self.player_hand) > self._get_hand_value(self.dealer_hand):
                reward = 3 #牌的点数大过庄家
            elif self._get_hand_value(self.player_hand) < self._get_hand_value(self.dealer_hand):
                reward = -1 #牌的点数小过庄家
            else:
                reward = 0
        return self._get_state(), reward, done

    def _deal_cards(self):
        random.shuffle(self.deck)#打乱牌堆
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]

    @staticmethod
    def _get_card_value(rank):
        if rank == "A":
            return 11
        elif rank in ["K", "Q", "J", "10"]:
            return 10
        else:
            return int(rank)

    def _get_hand_value(self, hand):#soft计算
        num_aces = sum(1 for card in hand if card[0] == "A")
        hand_value = sum(self._get_card_value(card[0]) for card in hand)
        while hand_value > 21 and num_aces > 0:
            hand_value -= 10
            num_aces -= 1
        return hand_value

    def _is_bust(self,hand):
        return self._get_hand_value(hand) > 21
    def _get_pokers_now(self):#未知的牌
        deck = [(rank, suit) for rank in self.ranks for suit in self.suits]
        state = np.zeros(4*13)
        for i in self.deck:
            state[deck.index((i[0],i[1]))]+=1
        #庄家有一张是暗牌
        state[deck.index((self.dealer_hand[1][0],self.dealer_hand[1][1]))]+=1
        return state

    def _get_pokers_vl(self,rank, suit):
        deck = [(rank, suit) for rank in self.ranks for suit in self.suits]
        return deck.index((rank, suit))
        
    def _get_state(self):
        
        state = np.zeros((4*13, 2))
        now_state=self._get_pokers_now()
        for i in range(len(self.player_hand)):
            rank, suit = self.player_hand[i]
            value = self._get_card_value(rank)
            if i == 0:
                state[self._get_pokers_vl(rank, suit)][0] += 1
                state[self._get_pokers_vl(self.dealer_hand[0][0],self.dealer_hand[0][1])][0] -=1
            else:
                state[self._get_pokers_vl(rank, suit)][1] += 1
        state=np.insert(state, 2, now_state, axis=1)
        state=state.flatten()
        
        state=np.round(state).astype(int).tolist()
        state = tuple(state)
        return state
    
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000000)
        self.gamma = 0.1    # 折扣因子（gamma）#0-1   越小越注重眼前利益

        self.epsilon = 1.0   # 探索率（epsilon）
        self.epsilon_decay = 0.995  # 探索率衰减系数
        self.epsilon_min = 0.01    # 最小探索率
        self.learning_rate = 0.001    # 学习率
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=None):
        # 使用 epsilon-贪婪策略选择动作
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            state_array = np.array([int(x) if isinstance(x, str) else x for x in state])
            state_array = state_array.astype(float)
            state_tensor = torch.from_numpy(state_array).float()
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def replay(self, batch_size):
        # 从经验回放缓冲区中随机抽取一批经验进行训练
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.from_numpy(next_state).float()
                target = reward + self.gamma * self.model(next_state_tensor).max().item()
            state_array = np.array([int(x) if isinstance(x, str) else x for x in state])   
            state_array = state_array.astype(float)
            state_tensor = torch.from_numpy(state_array).float()
            target_f = self.model(state_tensor).squeeze(0).data.numpy()
            target_f[action] = target
            target_f = torch.from_numpy(target_f).unsqueeze(0)
            self.model.train()
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.state_dict(), name)

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

    return prediction
def train_blackjack_dqn(n_episodes=10000, buffer_size=2000, batch_size=32,num_decks=1):
    # 创建 21 点游戏环境
    env = BlackjackEnv(num_decks)
    state_size = 156
    action_size = 2

    # 创建经验回放缓冲区对象和 DQN 智能体对象
    replay_buffer = deque(maxlen=buffer_size)
    agent = DQNAgent(state_size, action_size)

    # 训练 DQN 智能体
    for i_episode in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            if action!=0 or action!=1:
                action = agent.act(state)
            next_state, reward, done= env.step(action)
            if done:
                # 如果玩家赢了，则奖励为 +1；如果玩家输了，则奖励为 -1
                reward = 3 if reward == 3 else -1
            else:
                reward = 0  # 如果游戏没有结束，则奖励为 0
            next_state_array = np.array([int(x) if isinstance(x, str) else x for x in next_state])
            agent.remember(state, action, reward, next_state_array, done)
            state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if (i_episode + 1) % 1000 == 0:
            print(f"Episode {i_episode + 1}/{n_episodes}")

        if (i_episode + 1) % 1000 == 0:

            agent.save('model/dqn_21_'+str(i_episode + 1)+'.pt')

    # 使用训练好的智能体玩 21 点游戏，并观察智能体做出的动作和获得的奖励
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done= env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        state = next_state

    return agent


trained_agent = train_blackjack_dqn(n_episodes=10000)

trained_agent.save('model/dqn_21.pt')

"""
这段 Python 代码实现了使用深度 Q 学习算法（DQN）来训练一个智能体玩 21 点游戏。具体来说，该程序定义了一个名为 `DQNAgent` 的类，它包含了一个神经网络模型、经验回放缓冲区以及一些方法来训练该模型。然后我们定义了一个名为 `train_blackjack_dqn` 的函数，在其中使用 OpenAI Gym 提供的 `Blackjack-v1` 环境来训练我们的神经网络模型。

在 `DQNAgent` 类中，我们实现了以下功能：

- 创建一个具有两个隐藏层的神经网络模型
- 记录经验到回放缓冲区中
- 使用 epsilon-贪婪策略选择动作
- 从经验回放缓冲区中随机抽取一批经验进行训练
- 加载和保存神经网络模型

在 `train_blackjack_dqn` 函数中，我们执行以下步骤：

- 创建 21 点游戏环境
- 创建经验回放缓冲区对象和 DQN 智能体对象
- 训练 DQN 智能体
- 使用训练好的智能体玩 21 点游戏，并观察智能体做出的动作和获得的奖励

在训练期间，我们记录了智能体每 1000 次迭代时的模型，并将其保存到文件中。训练完成后，使用训练好的智能体玩一次 21 点游戏，并输出智能体做出的动作和获得的奖励。

请注意，在代码中，我们对环境中的状态进行了一些预处理，将其转换为一个浮点数数组，并传递给神经网络模型。我们还使用了 PyTorch 框架来定义和训练神经网络。
"""
