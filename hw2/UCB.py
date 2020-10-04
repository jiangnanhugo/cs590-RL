import numpy as np
from hw2.bernoullibandit import BernoulliBandit
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.grid(None)

class Upper_Confidence_bound(object):
    def __init__(self, m, means, delta):
        self.m = m
        self.k = len(means)
        self.mu_hat = np.zeros(self.k)
        self.UCB = np.zeros(self.k)
        self.T = np.zeros(self.k)
        self.means = means
        self.delta = delta
        self.bandit = BernoulliBandit(means=self.means)

    def run_UCB(self, n):
        for t in range(n):
            a_t = np.argmax(self.UCB)
            reward_t = self.bandit.pull(a_t)
            for i in range(self.k):
                if self.T[i] == 0:
                    self.UCB[i] = 10 ** 10
                else:
                    self.UCB[i] = self.mu_hat[i] + np.sqrt(2 * np.log2(1/self.delta)/self.T[i])
            self.mu_hat[a_t]=(self.mu_hat[a_t]*self.T[a_t]+reward_t)/(self.T[a_t]+1)
            self.T[a_t] += 1




if __name__ == '__main__':
    K = 5
    deltas = [0.1,0.5,0.9]
    means = np.random.random(K)
    N = 1000
    ratio = [(i+1)/100 for i in range(10*10)]

    x, z = [],  [[] for i in range(len(deltas))]
    dt = 0
    x=[int(N * ra) for ra in ratio]
    for delta in deltas:
        for i, ra in enumerate(ratio):
            m = int(N * ra)
            sumed = 0
            for tryout in range(10):
                ucb_method = Upper_Confidence_bound(m, means, delta)
                ucb_method.run_UCB(N)
                sumed +=ucb_method.bandit.random_regret()
            z[dt].append(sumed /10)
        dt += 1

    plt.plot(x, z[0],'r',label="delta(0.1)")
    print(x)
    print(z[0])
    plt.plot(x, z[1],'b',label="delta(0.5)")
    plt.plot(x, z[2],'g',label="delta(0.9)")

    plt.xlabel("the raio of N/m")
    plt.ylabel("The regret")
    # plt.show()
    plt.savefig("ucb.pdf")
