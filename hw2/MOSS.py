import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.grid(None)
from hw2.bernoullibandit import BernoulliBandit

def log_star(x):
    maxed = np.max([1,x])
    return np.log2(maxed)

# page 122
class Minimax_optimal_Strategy_in_stochastic_case(object):
    def __init__(self, m, means):
        self.m = m
        self.K = len(means)
        self.mu_hat = np.zeros(self.K)
        self.estimate = np.zeros(self.K)
        self.T = np.zeros(self.K)
        self.means = means
        self.bandit = BernoulliBandit(means=self.means)

    def run_Moss(self, n):
        for t in range(n):
            for i in range(self.K):
                if self.T[i] == 0:
                    self.estimate[i] = 10 ** 10
                else:
                    self.estimate[i] = self.mu_hat[i] + np.sqrt(4*log_star(N/self.K * self.T[i])/self.T[i])
            a_t = np.argmax(self.estimate)
            reward_t = self.bandit.pull(a_t)

            self.mu_hat[a_t] = (self.mu_hat[a_t] * self.T[a_t] + reward_t)/(self.T[a_t] + 1)
            self.T[a_t] += 1


if __name__ == '__main__':
    K = 5
    means = np.random.random(K)
    N = 1000
    ratio = [(i+1)/100 for i in range(10*10)]

    x, y = [], []
    for i, ra in enumerate(ratio):
        m = int(N * ra)
        sumed = 0
        for tryout in range(10):
            moss_method = Minimax_optimal_Strategy_in_stochastic_case(m, means)
            moss_method.run_Moss(N)
            sumed+=moss_method.bandit.random_regret()
        x.append(ratio[i])
        y.append(sumed / 10)
            # same plotting code as above!
    plt.plot(x, y)

    plt.xlabel("the raio of N/m")
    plt.ylabel("The regret")
    # plt.show()
    plt.savefig("moss.pdf")
