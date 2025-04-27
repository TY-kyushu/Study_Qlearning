import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class State():
#この State クラスは、グリッド上の状態（マス目の位置）を表すためのクラスです。
    def __init__(self,row=-1,column=-1):#row:行番号(デフォルトが-1),column:列番号(デフォルトが-1)
        self.row=row
        self.column=column

    def __repr__(self):#オブジェクトの文字列表現を返す関数
        return "<State:[{},{}]>".format(self.row,self.column)
    
    def clone(self):#現在の状態をコピーする関数
        return State(self.row,self.column)
    
    def __hash__(self):#これはハッシュ関数。State オブジェクトを辞書のキーや集合(set)の要素として使えるようにするために必要です。
        return hash((self.row,self.column))
    
    def __eq__(self,other):#等価比較（==）のための関数
        return self.row == other.row and self.column == other.column
    
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
#グリッド型の環境（マス目状のフィールド）を定義して、エージェントが移動・行動・報酬を得るシミュレーションを行うためのもの
class Environment():

    def __init__(self, grid, move_prob=1.0):
        # grid は二次元配列. その値は属性として扱われます.
        # 各属性は以下の通り
        #  0: ordinary cell(通れるマス)
        #  -1: damage cell (到達すると終了するマス)
        #  1: reward cell (到達すると終了するマス)
        #  9: block cell (通れないマス)
        self.grid = grid
        #現在のエージェントの位置
        self.agent_state = State()

        # 通常マスの報酬はマイナスにするする
        # これはより早くゴールするようにするため
        self.default_reward = -0.04

        # move_probはエージェントが意図した方向に動ける確率.
        # つまり意図しない方向に移動する確率は(1 - move_prob).
        self.move_prob = move_prob
        self.reset()

#「@property」はメソッドを"プロパティ（属性）っぽく"使えるようにするためのデコレーター
#つまり関数を変数っぽく表すもの
#env.row_length()→env.row_length

    #行を数える
    @property
    def row_length(self):
        return len(self.grid)

    #列を数える
    @property
    def column_length(self):
        return len(self.grid[0])

    #行動を返す
    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    #有効なマス(9以外)の位置情報を返す
    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    # def transit_func(self, state, action):
    #     transition_probs = {}
    #     if not self.can_action_at(state):
    #         # Already on the terminal cell.
    #         return transition_probs

    #     opposite_direction = Action(action.value * -1)

    #     for a in self.actions:
    #         prob = 0
    #         if a == action:
    #             prob = self.move_prob
    #         elif a != opposite_direction:
    #             prob = (1 - self.move_prob) / 2

    #         next_state = self._move(state, a)
    #         if next_state not in transition_probs:
    #             transition_probs[next_state] = prob
    #         else:
    #             transition_probs[next_state] += prob

    #     return transition_probs

    #0or-1かを判断するかを判断する
    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0 or -1:
            return True
        else:
            return False

    #指定されたstateで動けない方向をチェックし、可能な行動を返す
    def can_action(self, state, actions):
        can_actions = []
        # 壁かどうか確かめる.
        if self.grid[state.row -1][state.column] != 9:
            can_actions.append(0)
        if self.grid[state.row +1][state.column] != 9:
            can_actions.append(1)
        if self.grid[state.row][state.column -1] != 9:
            can_actions.append(2)
        if self.grid[state.row][state.column +1] != 9:
            can_actions.append(3)
        return can_actions

    #動いた先のstateを返す
    def _move(self, state, action):
        if not self.can_action_at(state):
            print(state.row, state.column)
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == 0:
            next_state.row -= 1
        elif action == 1:
            next_state.row += 1
        elif action == 2:
            next_state.column -= 1
        elif action == 3:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    #いる場所に応じて報酬を返す
    #終了判定をdoneで行う
    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == -1:
            # Get reward! and the game ends.
            reward = 1
            done = True

        return reward, done

    #エージェントのいる位置を初期位置にリセット
    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(1, 1)
        return self.agent_state

    #transit()をもとに実際に1step実行し、次の状態・報酬・終了判定を返す
    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    #遷移先と報酬、終了判定をシミュレート
    def transit(self, state, action):
        next_state = self._move(state, action)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

#ε-greedy方策の記録・可視化
class ELAgent():

    def __init__(self, epsilon):
        #行動価値関数
        self.Q = {}
        #探索の割合
        self.epsilon = epsilon
        #報酬記録用リスト
        self.reward_log = []

    #行動を選択する
    def policy(self, s, actions):
        #ε内であればランダムに行動を選択
        if np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]
        #ε外であれば行動価値関数が最大の方策を選択
        #状態ｓが未登録であればランダムに行動を選択
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return actions[np.random.randint(len(actions))]

    #報酬記録をリセット
    def init_log(self):
        self.reward_log = []

    #報酬を記録
    def log(self, reward):
        self.reward_log.append(reward)

    #報酬の記録をグラフにする
    def show_reward_log(self, interval=25, episode=-1):
        #直近25回分の報酬の平均とばらつきを表示
        if episode > 0:
            #[-interval:]はリストの最後から取り出す書き方
            rewards = self.reward_log[-interval:]
            #平均と分散を小数点以下３桁で
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print('At Episode {} average reward is {} (+/-{})'.format(episode, mean, std))
        
        #全体をグラフで見る
        else:
            #報酬の履歴を区切る位置を記録
            indices = list(range(0, len(self.reward_log), interval))
            #それぞれの区間で平均と分散を記録
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            #グラフにプロット
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title('Step History')
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color='g')
            plt.plot(indices, means, 'o-', color='g', label='Rewards for each {} episode'.format(interval))
            plt.legend(loc='best')
            #自動保存
            plt.savefig('Step History_{}.png'.format(self.epsilon))
            plt.show()

#Q学習の実装
class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.20):
        #親クラスのメソッドを呼び出すための関数
        super().__init__(epsilon)

    #learn(self, env, episode_count=500, gamma=0.9, learning_rate=0.1, render=False, report_interval=50)
    def learn(self, env, episode_count=500, gamma=0.9, learning_rate=0.1, report_interval=50):
        #報酬記録とQテーブルの初期化
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(4))
        for e in range(episode_count):
            #envはEnvironmentクラス内で定義した関数を使うときに付ける
            #下のほうでenv = Environment(grid)でインスタンスを作成している
            s = env.reset()
            done = False
            count = 0
            while not done:
                #renderはopenAIGymに用意されている関数なので取り除く
                #if render:
                    #env.render()
                can_actions = env.can_action(s, actions)
                a = self.policy(s, can_actions)
                n_state, reward, done = env.step(a)
                #gain=新しく得た行動状態価値関数
                gain = reward + gamma * (max(self.Q[n_state]))
                #現在のQ値
                estimated = self.Q[s][a]
                #Q値を更新する
                self.Q[s][a] += learning_rate * (gain - estimated)
                #ステップ数をカウント
                s = n_state
                count += 1

            #whileにかかるelse
            else:
                #ステップ数を記録
                self.log(count)

            #指定された間隔ごとにログを表示
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    grid = [
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 9],
    [9, 0, 0, 9, 0, 9, 0, 0, 9, 9, 0, 9],
    [9, 9, 0, 9, 0, 9, 9, 9, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9],
    [9, 9, 9, 9, 9, 0, 9, 0, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 0, 9, 0, 9, 0, 0, 9],
    [9, 0, 9, 9, 9, 9, 9, 0, 9, 0, 0, 9],
    [9, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 9],
    [9, 0, 0, 9, 0, 0, 0, 0, 9, 0, -1, 9],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]




    env = Environment(grid)
    agent = QLearningAgent()
    agent.learn(env, episode_count=500)
    agent.show_reward_log()


#このファイルが直接実行されたときだけ train() を呼び出すモノ
#'__main__'は「スクリプトとして直接実行」
if __name__ == '__main__':
    train()