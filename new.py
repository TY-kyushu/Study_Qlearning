import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class State():
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State:[{},{}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Environment():

    def __init__(self, grid, move_prob=1.0):
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0 or -1:
            return True
        else:
            return False

    def can_action(self, state, actions):
        can_actions = []
        if self.grid[state.row - 1][state.column] != 9:
            can_actions.append(0)
        if self.grid[state.row + 1][state.column] != 9:
            can_actions.append(1)
        if self.grid[state.row][state.column - 1] != 9:
            can_actions.append(2)
        if self.grid[state.row][state.column + 1] != 9:
            can_actions.append(3)
        return can_actions

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        if action == 0:
            next_state.row -= 1
        elif action == 1:
            next_state.row += 1
        elif action == 2:
            next_state.column -= 1
        elif action == 3:
            next_state.column += 1

        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward
        done = False
        attribute = self.grid[state.row][state.column]
        if attribute == -1:
            reward = 1
            done = True
        return reward, done

    def reset(self):
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


class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return actions[np.random.randint(len(actions))]

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=25, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print('At Episode {} average reward is {} (+/-{})'.format(episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title('Step History')
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color='g')
            plt.plot(indices, means, 'o-', color='g', label='Rewards for each {} episode'.format(interval))
            plt.legend(loc='best')
            plt.savefig('Step History_{}.png'.format(self.epsilon))
            plt.show()


class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.20):
        super().__init__(epsilon)

    def learn(self, env, episode_count=500, gamma=0.9, learning_rate=0.1, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(env.actions))
        actions = list(range(4))
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                can_actions = env.can_action(s, actions)
                a = self.policy(s, can_actions)
                n_state, reward, done = env.step(a)
                gain = reward + gamma * (max(self.Q[n_state]))
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                
                # 各マスごとのQ値を表示
                self.display_q_values(env)

                s = n_state

            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

    def display_q_values(self, env):
        # Q値をグリッドに数字で表示
        print("Q-values at current step:")
        for row in range(env.row_length):
            for col in range(env.column_length):
                state = State(row, col)
                if state in self.Q:
                    # 各状態ごとのQ値の最大値を表示
                    print(f"({row}, {col}): {max(self.Q[state]):.2f}", end="  ")
                else:
                    print("  -    ", end="  ")
            print()
        input()

def train():
    grid = [
        [9, 9, 9, 9, 9],
        [9, 0, 0, 0, 9],
        [9, 0, 9, 0, 9],
        [9, 9, 0, -1, 9],
        [9, 9, 9, 9, 9]
    ]

    env = Environment(grid)
    agent = QLearningAgent()
    agent.learn(env, episode_count=500)


if __name__ == '__main__':
    train()
