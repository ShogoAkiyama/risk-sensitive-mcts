from __future__ import print_function
from common import *
from scipy import stats
#from sets import Set

class MultiArmedBandit(MDP):
    def __init__(self, param=0):
        self.param = param

    # reset environment and return s0
    def reset(self):
        return 0

    # return a scipy.stats like distribution p(sp|s,a)
    def transition_func(self, s, a):
        probs = [1., 0., 0.]
        if a == 0:
            probs = [0., 0.2, 0.8]
        elif a == 1:
            if self.param == 0:
                probs = [0., 0.18, 0.82]
            elif self.param == 1:
                probs = [0., 0.5, 0.5]
        elif a == 2:
            if self.param == 0:
                probs = [0., 0.14, 0.86]
            elif self.param == 1:
                probs = [0., 0.66, 0.34]
        elif a == 3:
            if self.param == 0:
                probs = [0., 0., 1.]
            elif self.param == 1:
                probs = [0., 1., 0.]

        return stats.rv_discrete(name='Tsa', values=(self.state_space, probs), seed=None)

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        rewards = [0,1,0]
        return rewards[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s != 0

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return [0,1,2]

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        return [0,1,2,3]

class GridWorldTricky(MDP):
    def __init__(self, param=0):
        self.param = param
        #self.terminal_set = Set([-1, 6, 12, 18])
        self.terminal_set = set([-1, 4,5,6, 11,12,13, 18,19,20])

    def s_to_xy(self, s):
        return (s % 7, int(s / 7))

    def xy_to_s(self, x, y):
        if x < 0 or x >= 7:
            return -1
        if y < 0 or y >= 7:
            return -1
        return 7*y + x

    # reset environment and return s0
    def reset(self):
        return self.xy_to_s(3,3)

    # return a scipy.stats like distribution p(sp|s,a)
    def transition_func(self, s, a):
        x,y = self.s_to_xy(s)
        xp = []
        yp = []
        w = []
        if a == 0: # UP
            xp = [x]
            yp = [y+1]
            w = [1.]
        elif a == 1: # RIGHT
            xp = [x+(self.param+1)]
            yp = [y]
            w = [1.]
        elif a == 2: # DOWN
            xp = [x]
            yp = [y-1]
            w = [1.]
        elif a == 3: # LEFT
            xp = [x-(self.param+1)]
            yp = [y]
            w = [1.]

        sp_v = [self.xy_to_s(x_,y_) for (x_,y_) in zip(xp, yp)]

        # combine duplicate off-map states
        filtered_sp_v = [-1]
        filtered_w = [0.]
        for i in range(len(sp_v)):
            if sp_v[i] == -1:
                filtered_w[0] += w[i]
            else:
                filtered_sp_v.append(sp_v[i])
                filtered_w.append(w[i])

        return stats.rv_discrete(name='Tsa', values=(filtered_sp_v, filtered_w), seed=None)

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        if sp == -1:
            return -1
        h = 1.
        l = 0.2
        reward = [-h, -h, -h,  0, -h, -h, +h, \
                  -h, -h, -h,  0, -h, +h, -h, \
                  -h, -h, -h,  0, +h, -h, -h, \
                  -l, -l, -l,  0, -l, -l, -l, \
                  -l, -l,  0,  0,  0, -l, -l, \
                  -l,  0,  0,  0,  0,  0, -l, \
                   0,  0,  0,  0,  0,  0,  0]

        return reward[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s in self.terminal_set

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return np.arange(7*7)

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        return np.arange(4)

    def render(self, s):
        for st in self.state_space:
            if st == s:
                print(".", end='')
            else:
                print("@", end='')
            if st % 7 == 6:
                print("")

class GridWorld(MDP):
    def __init__(self, param=0):
        self.param = param
        #self.terminal_set = Set([-1, 6, 12, 18])
        self.terminal_set = { -1, 3,4, 8,9, 13,14 }

    def s_to_xy(self, s):
        return (s % 5, int(s / 5))

    def xy_to_s(self, x, y):
        if x < 0 or x >= 5:
            return -1
        if y < 0 or y >= 3:
            return -1
        return 5*y + x

    # reset environment and return s0
    def reset(self):
        return self.xy_to_s(0,2)

    # return a scipy.stats like distribution p(sp|s,a)
    def transition_func(self, s, a):
        x,y = self.s_to_xy(s)
        xp = []
        yp = []
        w = []
        if a == 0: # UP
            xp = [x]
            yp = [y+1]
            w = [1.]
        elif a == 1: # RIGHT
            xp = [x+(self.param+1)]
            yp = [y]
            w = [1.]
        elif a == 2: # DOWN
            xp = [x]
            yp = [y-1]
            w = [1.]
        elif a == 3: # LEFT
            xp = [x-(self.param+1)]
            yp = [y]
            w = [1.]

        sp_v = [self.xy_to_s(x_,y_) for (x_,y_) in zip(xp, yp)]

        # combine duplicate off-map states
        filtered_sp_v = [-1]
        filtered_w = [0.]
        for i in range(len(sp_v)):
            if sp_v[i] == -1:
                filtered_w[0] += w[i]
            else:
                filtered_sp_v.append(sp_v[i])
                filtered_w.append(w[i])

        return stats.rv_discrete(name='Tsa', values=(filtered_sp_v, filtered_w), seed=None)

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        if sp == -1:
            return -1
        h = 1.
        m = 0.8
        l = 0.1
        reward = [-l, -l, -l, +h, +h, \
                  -l, -l, -m, +h, +h, \
                  -l, -m, -m, +h, +h]

        return reward[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s in self.terminal_set

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return np.arange(5*3)

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        return np.arange(4)

    def render(self, s):
        for st in self.state_space:
            if st == s:
                print(".", end='')
            else:
                print("@", end='')
            if st % 5 == 4:
                print("")


class LavaGoalOneD(MDP):
    def __init__(self, param=0):
        self.param = param
        self.terminal_set = { -1,0, 6 }
        # self.terminal_set = { -1, 6 }
        # each 2d array corresponds to a given parameter setting
        # each columnn corresponds to moving [-2, -1, 0, 1, 2]
        # each row corresponds to probs under every action
        self.transition_probs = [ [[0.9, 0.1, 0, 0, 0],
                                   [0, 0.9, 0.1, 0, 0],
                                   [0, 0, 0.1, 0.9, 0],
                                   [0, 0, 0, 0.1, 0.9]],
                                  [[0, 0.5, 0.5, 0, 0],
                                   [0.5, 0.5, 0, 0, 0],
                                   [0, 0, 0, 0.5, 0.5],
                                   [0, 0, 0.5, 0.5, 0]],
                                  [[0, 0, 0, 0.1, 0.9],
                                   [0, 0, 0.1, 0.9, 0],
                                   [0, 0.9, 0.1, 0, 0],
                                   [0.9, 0.1, 0, 0, 0]],
                                  [[0, 0, 0.5, 0.5, 0],
                                   [0, 0, 0, 0.5, 0.5],
                                   [0.5, 0.5, 0, 0, 0],
                                   [0, 0.5, 0.5, 0, 0]] ]


    # reset environment and return s0
    def reset(self):
        return 2

    # return a scipy.stats like distribution p(sp|s,a)
    def transition_func(self, s, a):
        sp = [max(0,min(6,s+k)) for k in [-2,-1,0,1,2]]
        w = self.transition_probs[self.param][a]
        return stats.rv_discrete(name='Tsa', values=(sp, w), seed=None)

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        # reward = [-2, -0.1, -0.1, -0.1, -0.1, -0.1, +1.]
        reward = [-2., -0.1, 0, 0.2, 0.5, 0.7, 1.0]

        return reward[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s in self.terminal_set

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return np.arange(7)

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        return np.arange(4)

    def render(self, s):
        string = "X-----G"
        print(string[:s] + "*" + string[s+1:])
