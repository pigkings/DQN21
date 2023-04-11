# usr/bin/env python3;
#-*-coding: utf-8-*
import random
from sys import exit

class Poke:
    '''
    Poke类用来初始化一个牌堆
    '''
    def __init__(self):
        self.cards = [[face, suite] for face in "♠♥♦♣" for suite in [1,2,3,4,5,6,7,8,9,10,'J','Q','K']]
        random.shuffle(self.cards)


class Dealer:
    '''
    Dealer类初始化一个荷官
    主要用来实现取牌和发牌的作用
    '''
    def __init__(self):
        self.cards = Poke().cards

    def give_one_card(self):
        '''
        给玩家发牌
        return: list
        '''
        if not self.cards:
            # 重新取一副牌并洗牌
            self.cards.extend(Poke().cards)
        return self.cards.pop()


class Player:
    def __init__(self, name):
        '''
        初始化实例属性
        '''
        self.name = name
        self.score = 0
        self.points = 0
        self.cards_in_hand = []

    def init(self):
        '''
        重置计数器和牌列表
        '''
        self.cards_in_hand = []
        self.points = 0

    def now_count(self):
        '''
        更新计数器
        '''
        point = 0 
        for face, suite in self.cards_in_hand:
            if suite in ['J', 'Q', 'K']:
                suite = 10
            point += suite
        # 判断是否有A，如果有A再判断是否大于11，如果是的话A当做1，否的话当做11
        for card in self.cards_in_hand:
            if card[1] == 1 and point + 10 < 21:
                self.points = point + 10
            else:
                self.points = point

    def is_win(self, player):
        '''
        未提前结束回合时，判断玩家输赢
        param: player 进行比较的玩家
        '''
        s1 = self.points
        s2 = player.points
        if s1 > s2:
            print(f"玩家{self.name}点数为{s1}, 电脑{player.name}点数为{s2}, 玩家{self.name}赢了！")
            self.score += 1
        elif s1 == s2:
            print(f"玩家{self.name}点数为{s1}, 电脑{player.name}点数为{s2}, 平局！")
        else:
            print(f"玩家{self.name}点数为{s1}, 电脑{player.name}点数为{s2}, 电脑{player.name}赢了！")
            player.score += 1            

    def get(self, *cards):
        '''
        玩家取荷官发的牌，并更新计数器
        param： *cards 一个或多个list类型表示的牌
        '''
        for card in cards:
            self.cards_in_hand.append(card)
        self.now_count() # 重置分数


def main(dealer: Dealer, computer: Player, human: Player):
    '''
    游戏控制主函数
    '''
    # 回合数
    count = 0
    try:
        while True:
            count += 1
            print(f"第{count}轮比赛开始：")
            # 设置提前结束标志
            flag = False
            # 新回合初始化计数器和牌
            human.init()
            computer.init()
            # 准备发牌，两张牌给human, 两张牌给电脑{computer.name}
            human.get(dealer.give_one_card(), dealer.give_one_card())
            computer.get(dealer.give_one_card(), dealer.give_one_card())
            print(f"玩家{human.name}手中的牌是{human.cards_in_hand[-2]}, {human.cards_in_hand[-1]}")
            print(f"电脑{computer.name}手中的牌是{computer.cards_in_hand[-2]}, ?")
            # 判断是否21点
            if human.points == 21 == computer.points:
                print("玩家{human.name}和电脑{computer.name}都为21点，平局！")
            elif human.points == 21:
                print("玩家{human.name}的点数为21点，恭喜玩家{human.name}赢了！")
                human.score += 1
            else:
                # 玩家要牌
                while True:
                    if_next_card = input("是否继续要牌：(Y/N)")
                    if if_next_card in ['N', 'n']:
                       break                   
                    elif if_next_card in ['Y', 'y','']:
                        human.get(dealer.give_one_card())
                        print(f"玩家{human.name}得到一张{human.cards_in_hand[-1]}, 玩家{human.name}手中的牌是{human.cards_in_hand}")
                        # 判断玩家是否超过21点，如果是提前结束标志设置为True
                        if human.points > 21:
                            print(f"玩家{human.name}的点数{human.points}超过了21点,玩家{human.name}输了！")
                            computer.score += 1
                            flag = True
                            break
                # 电脑要牌
                if not flag:
                    # 电脑要牌逻辑，只要小于玩家分数就要牌
                    while computer.points < human.points:
                        computer.get(dealer.give_one_card())
                        print(f"电脑{computer.name}得到一张{computer.cards_in_hand[-1]}, 电脑{computer.name}手中的牌是{computer.cards_in_hand}")
                    # 先判断电脑是否大于21点，如果大于21点提前结束
                    if computer.points > 21:
                        print(f"电脑{computer.name}的点数为{computer.points}超过21点，恭喜玩家{human.name}赢了！")
                        human.score += 1
                    else:
                        # 没有提前结束也就是都小于21点的情况下，判断大小输赢
                        human.is_win(computer)
            print("-" * 30)
            # 是否进行下一局
            if_play_again = input("是否进行下一局:(Y/N)")
            # 如果继续，先打印总比分，再重新开始
            if if_play_again in ['Y', 'y','']:
                print(f"玩家{human.name}，电脑{computer.name}总比分为{human.score}:{computer.score}")
            # 如果停止，打印总比分并判处胜者后退出
            elif if_play_again in ['N', 'n']:
                print(f"玩家{human.name}，电脑{computer.name}总比分为{human.score}:{computer.score}")
                if human.score > computer.score:
                    print(f"{human.name}胜出！")
                elif human.score < computer.score:
                    print(f"{computer.name}胜出！")
                else:
                    print("战况激烈，你们打平了！")
                print("游戏结束")
                exit(0)
            else:
                print("输入有误，请重新输入：")
    except Exception:
        print("有bug，游戏结束！")

if __name__ == '__main__':
    computer = Player('Robot')
    human = Player('Lihaoer')
    dealer = Dealer()
    main(dealer, computer, human)
