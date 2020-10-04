import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.grid(None)
from hw2.bernoullibandit import BernoulliBandit
import seaborn as sns
sns.set()

class Explore_then_commit(object):
    def __init__(self, m, means):
        self.m = m
        self.K = len(means)
        self.mu_hat = np.zeros(self.K)
        self.T = np.zeros(self.K)
        self.means = means
        self.bandit = BernoulliBandit(means=self.means)

    def run_ETC(self, n):
        for t in range(n):
            if t <= self.m * self.K:
                a_t = (t % self.K)
            else:
                a_t = np.argmax(self.mu_hat)
            reward_t = self.bandit.pull(a_t)
            if t <= self.m * self.K:
                self.mu_hat[a_t] = (self.mu_hat[a_t] * self.T[a_t] + reward_t)/(self.T[a_t] + 1)
                self.T[a_t] += 1

if __name__ == '__main__':
    K = 5
    means = np.random.random(K)
    N = 1000
    ratio = [(i+1)/100 for i in range(5*10)]
    x, y = [], []
    for i, ra in enumerate(ratio):
        m = int(N * ra)
        sumed = 0
        for tryout in range(10):
            etc_method = Explore_then_commit(m, means)
            etc_method.run_ETC(N)
            sumed += etc_method.bandit.random_regret()
        x.append(ratio[i])
        y.append(sumed/ 10)


    # same plotting code as above!
    plt.plot(x, y)

    plt.xlabel("the raio of N/m")
    plt.ylabel("The regret")
    # plt.show()
    plt.savefig("etc.pdf")

    # plt.legend('ABCDEF', ncol=2, loc='upper left');



