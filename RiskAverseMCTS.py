from __future__ import print_function
from common import *
from scipy import optimize
from scipy.stats import rv_discrete
import scipy.linalg as la
import numpy as np
from copy import deepcopy

class RiskAverseMCTS(Agent):
    def __init__(self, state_space, action_space, mdps, belief, max_depth=1, max_r=1, alpha=1.0, n_iter=200):
        super(RiskAverseMCTS, self).__init__(state_space, action_space)
        self.mdps = deepcopy(mdps)
        self.n_mdps = len(mdps)
        self.orig_belief = belief
        self.belief = np.array(belief)
        self.N_belief_updates = 0
        self.adversarial_belief = np.array(belief)
        self.adversarial_belief_avg = np.array(belief)
        self.adversarial_dist = rv_discrete(values=(np.arange(self.n_mdps), self.adversarial_belief))
        self.action_space = action_space
        self.gamma = 0.9
        self.Nh = {}
        self.Nha = {}
        self.Qha = {}
        self.model_values = np.zeros(self.n_mdps)
        self.laplace_smoothing = 0
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)
        self.max_depth = max_depth
        self.max_r = max_r
        self.c = 2.0#max_r*max_depth + 0.0000001 # For finite horizon mdps. For infinite horizon, c > max_r/(1-gamma)
        self.eta = 0.05 # smoothing of distribution updates
        self.eta_agent = 0.3
        self.alpha = alpha
        self.K = 10
        self.n_iter = n_iter

    def reset(self):
        self.belief = np.array(self.orig_belief)
        self.N_belief_updates = 0
        self.adversarial_belief = np.array(self.orig_belief)
        self.adversarial_dist = rv_discrete(values=(np.arange(self.n_mdps), self.adversarial_belief))
        self.Nh = {}
        self.Nha = {}
        self.Qha = {}
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

    def action(self, s):
        self.MCTS(s)
        bestq = -np.inf
        besta = -1
        for a in self.action_space:
            qval = self.Qha[(s,a)]
            if qval > bestq:
                bestq = qval
                besta = a

        return besta
        # return self.avg_action((s,))

    def observe(self, s, a, r, sp):
        self.update_belief(s,a,sp)
        #print("belief:", self.belief)
        #pass

    def update_belief(self,s,a,sp):
        probs = [mdp.transition_func(s,a).pmf(sp) for mdp in self.mdps]
        belief = self.belief*probs
        self.belief = belief/np.sum(belief)
        self.adversarial_belief = deepcopy(self.belief)

    def MCTS(self, s):
        # reset tree from previous computation?
        self.adv_dists = []
        self.agent_est_value = []
        self.adv_est_value = []

        self.Nh = {}
        self.Nha = {}
        self.Qha = {}
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

        for itr in range(self.n_iter):
            # if itr % 10 == 0:
            #     print("\rItr", itr, ": Adversarial Belief:", self.adversarial_belief, "\tQ:", self.Qha, end='', flush=True)

            self.update_adversarial_belief()
            for k in range(self.K):
                mdp_i = self.adversarial_dist.rvs()
                R = self.simulate( (s,), mdp_i, 0 )

                self.model_counts[mdp_i] += 1
                self.model_values[mdp_i] += (R - self.model_values[mdp_i])/self.model_counts[mdp_i] # TODO double check this math

            # self.adv_dists.append(deepcopy(self.adversarial_belief_avg))
            # value = np.max([self.Qha[(s,a)] for a in self.action_space])
            # self.agent_est_value.append(value)
            # self.adv_est_value.append(np.dot(self.model_values, self.adversarial_belief_avg))
        # print("Q(h+a):\t", [(a,self.Qha[(s,a)]) for a in self.action_space])
        # print("N(h+a):\t", self.Nha)
        # print("V(theta):\t", self.model_values)
        # print("b_tilde(theta):\t", self.adversarial_belief)
        # print(np.dot(self.model_values, self.adversarial_belief))


    def update_adversarial_belief(self):
        Aeq = np.ones((1,self.n_mdps))
        beq = 1
        A1 = np.eye(self.n_mdps)
        b1 = self.belief*1/self.alpha
        A2 = -np.eye(self.n_mdps)
        b2 = np.zeros(self.n_mdps)
        A = np.vstack([A1,A2])
        b = np.vstack([b1, b2])

        # do we augment this with "lower confidence bounds?"
        # should we choose adversarial belief assuming the policy will do better or worse than the mean so far?
        # assuming the worst (i.e. optimism wrt the adversary) ensures exploration
        c = np.array(self.model_values)
        for i in range(self.n_mdps):
            if self.model_counts[i] != 0:
                c[i] -= self.c * np.sqrt(np.log(np.sum(self.model_counts))/self.model_counts[i])
            else:
                c[i] = -self.max_r
        res = optimize.linprog(c, A, b, Aeq, beq)

        self.adversarial_belief = res.x
        self.N_belief_updates += 1
        self.adversarial_belief_avg += (res.x - self.adversarial_belief_avg)/self.N_belief_updates

        mixed_strategy_belief = (1-self.eta)*self.adversarial_belief_avg + self.eta*res.x;
        self.adversarial_dist = rv_discrete(values=(np.arange(self.n_mdps), self.adversarial_belief))
        # print("V(theta):\t", self.model_values)
        # print("b_tilde(theta):\t", self.adversarial_belief)

    def simulate(self, h, mdp_i, depth, update_tree=True):
        if depth >= self.max_depth:
            return 0
        if self.mdps[mdp_i].done(h[-1]):
            return 0

        if h not in self.Nh:
            for a in self.action_space:
                self.Nha[h+(a,)] = 0
                self.Qha[h+(a,)] = 0

            a = self.sample_rollout_action(h)
            r, sp = self.mdps[mdp_i].step(h[-1], a)
            R = r + self.gamma*self.rollout(h + (a,sp), mdp_i, depth + 1)
            if update_tree:
                self.Nh[h] = 1
                self.Nha[h+(a,)] = 1
                self.Qha[h+(a,)] = R
            return R

        a = self.smooth_ucb_action(h)
        r, sp = self.mdps[mdp_i].step(h[-1], a)

        R = r + self.gamma*self.simulate(h + (a,sp), mdp_i, depth + 1)
        if update_tree:
            self.Nh[h] += 1
            self.Nha[h+(a,)] += 1
            self.Qha[h+(a,)] = self.Qha[h+(a,)] + (R - self.Qha[h+(a,)])*1./self.Nha[h+(a,)]
        return R

    def rollout(self, h, mdp_i, depth):
        if depth >= self.max_depth:
            return 0
        if self.mdps[mdp_i].done(h[-1]):
            return 0

        a = self.sample_rollout_action(h)
        r, sp = self.mdps[mdp_i].step(h[-1], a)
        return r + self.gamma*self.rollout(h + (a,sp), mdp_i, depth + 1)

    def sample_rollout_action(self, h):
        # randomly sample action
        return np.random.choice(self.action_space)

    def ucb_action(self, h):
        best_a = -1
        best_val = -np.inf
        for a in self.action_space:
            val = np.inf
            if self.Nha[h+(a,)] != 0:
                val = self.Qha[h+(a,)] + self.c * np.sqrt(np.log(self.Nh[h])/self.Nha[h+(a,)])

            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def avg_action(self, h):
        if h not in self.Nh:
            probs = np.ones(self.action_space)
        else:
            probs = np.array( [self.Nha[h+(a,)]*1. for a in self.action_space] )
        probs = probs/np.sum(probs)
        best_a = np.random.choice(self.action_space)
        return best_a

    def smooth_ucb_action(self, h):
        z = np.random.rand()
        best_a = -1
        if z < self.eta_agent:
            # return the action given by upper confidence bounds
            return self.ucb_action(h)
        else:
            return self.avg_action(h)

        return best_a
