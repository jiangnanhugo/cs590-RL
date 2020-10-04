import numpy as np
from hw2.bernoullibandit import BernoulliBandit
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.grid(None)

def f(t):
    return 1+ t* (np.log2(t))**2

eps = 1e-15
def d(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log2(x/y) + (1-x)* np.log2((1-x)/(1-y))


# page 122
class KL_UCB(object):
    def __init__(self, m, means):
        self.m = m
        self.k = len(means)
        self.mu_hat = np.zeros(self.k)
        self.estimate = np.zeros(self.k)
        self.T = np.zeros(self.k)
        self.means = means
        self.bandit = BernoulliBandit(means=self.means)

    def run_KL_UCB(self, n):
        for t in range(n):
            for i in range(self.k):
                if self.T[i] == 0:
                    self.estimate[i] = 10 ** 10
                else:
                    thresh = np.log2(f(t)) / self.T[i]
                    self.estimate[i] = -1
                    for j in range(100):
                        mu_tilde = j/100
                        tmp=d(self.mu_hat[i], mu_tilde)
                        if tmp<=thresh:
                            self.estimate[i] = mu_tilde

            a_t = np.argmax(self.estimate)
            reward_t = self.bandit.pull(a_t)

            self.mu_hat[a_t] = (self.mu_hat[a_t] * self.T[a_t] + reward_t) / (self.T[a_t] + 1)
            self.T[a_t] += 1

    def update_mean(self, a_t, reward_t):
        self.mu_hat[a_t] = (self.mu_hat[a_t] * self.T[a_t] + reward_t) / (self.T[a_t] + 1)


if __name__ == '__main__':
    K = 5
    means = np.random.random(K)
    N = 1000
    ratio = [(i + 1) / 100 for i in range( 100)]

    x, y = [], []
    for i, ra in enumerate(ratio):
        m = int(N * ra)
        sumed = 0
        for tryout in range(4):
            kl_ucb_method = KL_UCB(m, means)
            kl_ucb_method.run_KL_UCB(N)
            sumed += kl_ucb_method.bandit.random_regret()
        x.append(ratio[i])
        y.append(sumed / 4)
        print(ratio[i], sumed/4)
        # same plotting code as above!
    plt.plot(x, y)

    plt.xlabel("the raio of N/m")
    plt.ylabel("The regret")
    # plt.show()
    plt.savefig("klucb.pdf")
