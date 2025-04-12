import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class State():
#この State クラスは、グリッド上の状態（マス目の位置）を表すためのクラスです。
    def __init__(self,row=-1,column=-1):#row:行番号,column:列番号
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

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0 or -1:
            return True
        else:
            return False

    def can_action(self, state, actions):
        can_actions = []
        # Check whether the agent bumped a block cell.
        if self.grid[state.row -1][state.column] != 9:
            can_actions.append(0)
        if self.grid[state.row +1][state.column] != 9:
            can_actions.append(1)
        if self.grid[state.row][state.column -1] != 9:
            can_actions.append(2)
        if self.grid[state.row][state.column +1] != 9:
            can_actions.append(3)
        return can_actions


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

    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(1, 1)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        next_state = self._move(state, action)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

