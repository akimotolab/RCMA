import math
import numpy as np

class LmMa:
    def __init__(self, func, xmean0, sigma0):
        self.func = func
        self.N = len(xmean0)
        self.xmean = np.array(xmean0, copy=True)
        self.sigma = sigma0
        self.lam = 4 + int(3 * np.log(self.N))
        self.mu = self.lam // 2
        self.gamma = self.lam
        self.wi_raw = np.log(self.lam / 2 + 0.5) - np.log(np.arange(1, self.mu+1))
        self.wi = self.wi_raw / np.sum(self.wi_raw)
        self.mu_eff = 1.0 / np.sum(self.wi ** 2)
        self.c_s = 2.0 * self.lam / self.N
        self.sqrt_s = np.sqrt(self.c_s * (2.0 - self.c_s) * self.mu_eff)
        self.cp = self.lam / self.N * 4.0**(-np.arange(self.gamma))
        self.cd = 1.5**(-np.arange(self.gamma)) / self.N
        self.sqrt_cp = np.sqrt(self.cp * (2.0 - self.cp) * self.mu_eff)
        self.p_matrix = np.zeros((self.gamma, self.N))
        self.s = np.zeros(self.N)  ## HG initialized with ones wrongly!
        self.arf = np.empty(self.lam)
        self.t = 0
        self.neval = 0

    def onestep(self):
        arz = np.random.randn(self.lam, self.N)
        ary = np.copy(arz)
        for j in range(min(self.t, self.gamma)):
            ary = (1.0 - self.cd[j]) * ary + np.outer(np.dot(ary, self.p_matrix[j]), self.cd[j] * self.p_matrix[j])
        arx = self.xmean + self.sigma * ary
        arf = self.func(arx)
        self.neval += self.lam
        idx = np.argsort(arf)
        sum_z = np.dot(self.wi, arz[idx[:self.mu]])
        sum_d = np.dot(self.wi, ary[idx[:self.mu]])
        self.xmean += self.sigma * sum_d
        self.s = (1.0 - self.c_s) * self.s + self.sqrt_s * sum_z
        cp_column = self.cp.reshape((-1, 1))
        self.p_matrix = (1.0 - cp_column) * self.p_matrix + np.outer(cp_column, sum_z)
        self.sigma *= np.exp(0.5 * self.c_s * (np.dot(self.s, self.s) / self.N - 1.0))
        self.arf = arf
        self.t += 1

    @property
    def coordinate_std(self):
        return self.sigma