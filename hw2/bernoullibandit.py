import random


class BernoulliBandit(object):
    # Accepts a list of k >= 2 floats, each lying in [0, 1]
    def __init__(self, means):
         assert len(means) >= 2, "Requires at least 2 arms"
         self.means = means
         self.k = len(means)
         self.actions = []
         self.rewards = []

    # Accept a parameter 0 <= a <= K-1 and returns the
    # relaization of random variable X with P(X=1) being
    # the mean of the (a+1)-th arm
    def pull(self, a):
        assert 0 <= a <= self.k - 1, "0 <= a <= K - 1"
        self.actions.append(a)
        result = int(random.random() <= self.means[a])
        self.rewards.append(result)
        return result

    def random_regret(self):
        opt = len(self.actions) * max(self.means)
        random_regret = opt - sum(self.rewards)
        return random_regret

    def regret(self):
        opt = len(self.actions) * max(self.means)
        regret = opt - sum([self.means[a] for a in self.actions])
        return regret

