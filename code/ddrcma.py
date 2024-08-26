from collections import deque
from functools import partial
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def random_axes(dim, n_axes):
    R = np.random.normal(0, 1, (n_axes, dim))
    for i in range(n_axes):
        for j in range(i):
            R[i] = R[i] - np.dot(R[i], R[j]) * R[j]
        R[i] = R[i] / np.linalg.norm(R[i])
    return R


def myeigh(A, U, mode=-1, value_only=False):
    if (mode == -1 and A.shape[0] > A.shape[1]) or mode == 1:
        # based on QR decomposition, proposed in xCMA paper
        Q, R = np.linalg.qr(A)
        K = np.dot(R * U, R.T)
        if value_only:
            return np.linalg.eigvalsh(K)
        else:
            D, V = np.linalg.eigh(K)
            E = np.dot(Q, V)
            return D, E
    else:
        # standard
        if value_only:
            return np.linalg.eigvalsh(np.dot(A * U, A.T))
        else:
            return np.linalg.eigh(np.dot(A * U, A.T))


class AbstractModel:

    def __init__(self, N, lam, **kwargs):
        raise NotImplementedError

    @property
    def cumulation_factor(self):
        raise NotImplementedError

    @property
    def sqrt_condition_number(self):
        raise NotImplementedError

    @property
    def coordinate_std(self):
        raise NotImplementedError

    @property
    def sqrteigvals(self):
        raise NotImplementedError

    def get_axis(self, i):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def inverse_transform(self, x):
        raise NotImplementedError

    def update(self, sarz, zpc):
        raise NotImplementedError

    def decompose(self):
        raise NotImplementedError


class NullModel(AbstractModel):

    def __init__(self, N, lam):
        self.N = N
        self.lam = lam

    @property
    def cumulation_factor(self):
        return 1.0

    @property
    def sqrt_condition_number(self):
        return 1.0

    @property
    def coordinate_std(self):
        return np.ones(self.N)

    @property
    def sqrteigvals(self):
        return np.ones(self.N)

    def get_axis(self, i):
        ax = np.zeros(self.N)
        ax[i] = 1.
        return ax

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def update(self, sarz, zpc):
        pass

    def decompose(self):
        return 1.


class VSModel(AbstractModel):
    """Internal Covariance Model : VSModel
    
    The internal covariance matrix is expressed as
        C = In + V * (S - Ik) * V.T,
    where In and Ik are the identity matrix of dimension n and k, respectively, and
        S : k-by-k diagonal matrix representing the eigenvalues of C
        V : n-by-k matrix with orthonormal columns representing the eigenvectors of C
    """

    def __init__(self, N, lam,
                 kmax=None, flg_k_update=True, 
                 beta_cond=6.0, beta_wait=None, flg_active_update=True):

        self.N = N
        self.lam = lam
        self.beta_cond = beta_cond
        self.beta_wait = beta_wait if beta_wait else 10. * math.log10(beta_cond)
        self.flg_active_update = flg_active_update
        self.flg_k_update = flg_k_update

        # intermediate weights
        w = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam + 1))
        w[w > 0] /= np.sum(np.abs(w[w > 0]))
        w[w < 0] /= np.sum(np.abs(w[w < 0]))
        self.w = w
        self.mueff_positive = 1. / np.sum(w[w > 0] ** 2)
        self.mueff_negative = 1. / np.sum(w[w < 0] ** 2)

        # VS model parameters
        self.kmax = kmax if kmax is not None else lam
        self.S = np.ones(self.N)
        self.V = np.zeros((self.kmax, self.N))
        self.klong = 1 if self.flg_k_update else int(math.ceil(self.kmax / 2))
        self.kshort = 1 if self.flg_k_update else int(math.floor(self.kmax / 2))
        self.update_complexity(self.kshort, self.klong, force_reset=True)
        self.t_wait = 0

    def update_complexity(self, kshort, klong, force_reset=False):
        ks_old, kl_old = self.kshort, self.klong
        self.kshort, self.klong = kshort, klong
        if force_reset or (ks_old + kl_old != kshort + klong):
            # parameters for covariance matrix adaptation
            k = self.klong + self.kshort
            #expo = 0.75 #XXX
            expo = 1.0
            mu_prime = self.mueff_positive + 1. / self.mueff_positive - 2. + self.lam / (2. * self.lam + 10.)
            m = k * (2 * self.N + 1 - k) / 2 
            self.cone = 1. / (2 * (m / self.N + 1.) * (self.N + 1.) ** expo + self.mueff_positive / 2.)
            self.cmu = min(1. - self.cone, mu_prime * self.cone)
            self.cc = math.sqrt(self.mueff_positive * self.cone) / 2.
            self.wc = np.copy(self.w)
            self.wc[self.wc < 0] *= min(1., (1 - self.cone - self.cmu) / (self.N * self.cmu)) # Method 2 from DD-CMA paper
            # parameter for k adaptation
            self.t_wait_for_next_decrease = self.beta_wait / (self.cone + self.cmu * np.sum(np.abs(self.wc)))

    @property
    def cumulation_factor(self):
        return self.cc

    @property
    def sqrt_condition_number(self):
        mineig = self.S[0]
        maxeig = self.S[-1]
        return math.sqrt(maxeig / mineig)

    @property
    def coordinate_std(self):
        ks, kl = self.kshort, self.klong
        return np.sqrt(1.0 + np.dot(self.S[:ks] - 1.0, self.V[:ks] ** 2) + np.dot(self.S[-kl:] - 1.0, self.V[-kl:] ** 2))

    @property
    def sqrteigvals(self):
        return np.sqrt(self.S)

    def get_axis(self, i):
        k = self.klong + self.kshort
        ax = random_axes(self.N, 1)[0]
        if k == 0:
            return ax
        ii = i % k
        if ii < self.klong:
            if np.linalg.norm(self.V[-ii]) < 0.5:  # not yet set
                return ax
            else:
                return math.sqrt(self.S[-ii]) * self.V[-ii]
        else:
            ii = k - ii - 1
            if np.linalg.norm(self.V[ii]) < 0.5:  # not yet set
                return ax
            else:
                return math.sqrt(self.S[ii]) * self.V[ii]

    def transform(self, x):
        """y = (I + V*(S-I)*V^t)^{1/2} z
        Parameter
        ---------
        x : 1d or 2d array-like
            a vector or a list of vectors to be transformed
        """
        ks, kl = self.kshort, self.klong
        y = np.array(x, copy=True)
        if ks > 0:
            y += np.dot(np.dot(x, self.V[:ks].T) * (np.sqrt(self.S[:ks]) - 1.0), self.V[:ks])
        if kl > 0:
            y += np.dot(np.dot(x, self.V[-kl:].T) * (np.sqrt(self.S[-kl:]) - 1.0), self.V[-kl:])
        return y

    def inverse_transform(self, x):
        """z = (I + E*(S-I)*E^t)^{-1/2} * y
        sqrt(inv(I - EEt + ESEt))
        = sqrt(I - EEt + ES^-1Et)
        = I - EEt + E(S)^{-1/2}Et
        = I + E((S)^{-1/2} - I)Et

        Parameter
        ---------
        x : 1d or 2d array-like
            a vector or a list of vectors to be inversely transformed
        """
        ks, kl = self.kshort, self.klong
        y = np.array(x, copy=True)
        if ks > 0:
            y += np.dot(np.dot(x, self.V[:ks].T) * (1.0 / np.sqrt(self.S[:ks]) - 1.0), self.V[:ks])
        if kl > 0:
            y += np.dot(np.dot(x, self.V[-kl:].T) * (1.0 / np.sqrt(self.S[-kl:]) - 1.0), self.V[-kl:])
        return y

    def update(self, sarz, zpc):
        wc = self.wc
        # approximate C = fac * (I + V*S*Vt) + Z*W*Zt with gamma * (I + V*S*Vt)
        ks, kl = self.kshort, self.klong
        fac = 1.0 - (self.cmu * np.sum(wc) + self.cone)
        Z = np.column_stack((self.V[:ks].T,
                             self.V[-kl:].T,
                             self.transform(zpc),
                             self.transform(sarz)[wc > 0].T,
                             self.transform(sarz)[wc < 0].T))
        W = np.concatenate(((self.S[:ks] - 1.) * fac,
                            (self.S[-kl:] - 1.) * fac,
                            np.asarray([self.cone]),
                            self.cmu * wc[wc > 0],
                            self.cmu * self.N * wc[wc < 0] / np.linalg.norm(sarz[wc < 0], axis=1) ** 2))
        self.fac, self.Z, self.W = fac, Z, W

    def decompose(self):
        ks, kl = self.kshort, self.klong
        fac, Z, W = self.fac, self.Z, self.W

        Lam, U = myeigh(Z, W)
        lp = np.sum(Lam > 0.0)
        lm = np.sum(Lam < 0.0)
        lorth = max(kl - lp, 0) + max(ks - lm, 0)
        if lorth > 0:
            UU = U[:, Lam != 0.0]
            North = np.random.randn(self.N, lorth)
            North -= UU @ (UU.T @ North)
            Uorth, _ = np.linalg.qr(North)
        else:
            Uorth = np.empty((self.N, 0))
        tLam = np.ones(self.N) * fac
        tLam[:lm] += Lam[:lm]
        if lp > 0:
            tLam[-lp:] += Lam[-lp:]
        gamma = np.exp(np.mean(np.log(tLam[ks:self.N-kl])))

        self.S[:ks] = tLam[:ks] / gamma
        self.S[-kl:] = tLam[-kl:] / gamma
        self.S[ks:self.N-kl] = 1.0
        self.V[:min(ks, lm)] = U[:, :min(ks, lm)].T
        self.V[min(ks, lm):ks] = Uorth[:, :ks-min(ks, lm)].T
        if lp > 0:
            self.V[-min(kl, lp):] = U[:, -min(kl, lp):].T
        if kl > min(kl, lp):
            self.V[-kl:-min(kl, lp)] = Uorth[:, -(kl-min(kl, lp)):].T

        # Model Complexity Adaptation
        self.t_wait += 1    
        kshort, klong = ks, kl
        if self.flg_k_update:
            if self.S[ks-1] < 1.0/self.beta_cond:
                kshort = min(int(math.ceil(1.4 * self.kshort)), max(self.kmax // 2, self.kmax - klong))
                self.S[ks:kshort] = 1.0
                self.t_wait = 0
            elif self.t_wait > self.t_wait_for_next_decrease:
                kshort = np.sum(self.S < 1.0/self.beta_cond) + 1
                self.S[kshort:ks] = 1.0
            if self.S[-kl] > self.beta_cond:
                klong = min(int(math.ceil(1.4 * self.klong)), self.kmax - kshort)
                self.S[-kl:-klong] = 1.0
                self.t_wait = 0
            elif self.t_wait > self.t_wait_for_next_decrease:
                klong = np.sum(self.S > self.beta_cond) + 1
                self.S[-kl:-klong] = 1.0

            self.update_complexity(kshort, klong) 

        return gamma**0.5



class FullModel(AbstractModel):
    """Internal Covariance Model : Full Model"""

    def __init__(self, N, lam, beta_eig=None, flg_active_update=True):
        """
        Parameters
        ----------
        N : int
        lam : int
        beta_eig : float, optional (default = None)
            coefficient to control the frequency of matrix decomposition
        flg_active_update : bool, optional (default = True)

        """
        self.N = N
        self.lam = lam
        self.beta_eig = beta_eig if beta_eig else 10. * self.N
        self.flg_active_update = flg_active_update

        w = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam+1))
        w[w > 0] /= np.sum(np.abs(w[w > 0]))
        w[w < 0] /= np.sum(np.abs(w[w < 0]))
        self.mueff_positive = 1. / np.sum(w[w > 0] ** 2)
        self.mueff_negative = 1. / np.sum(w[w < 0] ** 2)
        
        # parameters for CMA (full model)
        self.C = np.eye(self.N)
        self.S = np.ones(self.N)
        self.B = np.eye(self.N)
        self.sqrtC = np.eye(self.N)
        self.invsqrtC = np.eye(self.N)
        self.Z = np.zeros((self.N, self.N))
        
        # parameters for covariance matrix adaptation
        expo = 0.75
        mu_prime = self.mueff_positive + 1. / self.mueff_positive - 2. + self.lam / (2. * self.lam + 10.)
        m = self.N * (self.N + 1) / 2
        self.cone = 1. / ( 2 * (m / self.N + 1.) * (self.N + 1.) ** expo + self.mueff_positive / 2.)
        self.cmu = min(1. - self.cone, mu_prime * self.cone)
        self.cc = math.sqrt(self.mueff_positive * self.cone) / 2.

        self.wc = w
        self.wc[w < 0] *= min(1. + self.cone / self.cmu, 1. + 2. * self.mueff_negative / (self.mueff_positive + 2.))

        self.teig = max(1, int(1. / (self.beta_eig * (self.cone + self.cmu))))
        self.t_stall = 0
        
    @property
    def cumulation_factor(self):
        return self.cc
    
    @property
    def sqrt_condition_number(self):
        return self.S.max() / self.S.min()

    @property
    def coordinate_std(self):
        return np.sqrt(np.diag(self.C))

    @property
    def sqrteigvals(self):
        return self.S

    def get_axis(self, i):
        return self.S[i] * self.B[:, i]

    def transform(self, x):
        return np.dot(x, self.sqrtC)

    def inverse_transform(self, x):
        return np.dot(x, self.invsqrtC)
        
    def update(self, sarz, zpc):
        wc = self.wc
        # Rank-mu
        if self.cmu == 0:
            rank_mu = 0.
        elif self.flg_active_update:
            rank_mu = np.dot(sarz[wc>0].T * wc[wc>0], sarz[wc>0]) - np.sum(wc[wc>0]) * np.eye(self.N)
            rank_mu += np.dot(sarz[wc<0].T * (wc[wc<0] * self.N / np.linalg.norm(sarz[wc<0], axis=1) ** 2),
                              sarz[wc<0]) - np.sum(wc[wc<0]) * np.eye(self.N)
        else:
            rank_mu = np.dot(sarz[wc>0].T * wc[wc>0], sarz[wc>0]) - np.sum(wc[wc>0]) * np.eye(self.N)
        # Rank-one
        if self.cone == 0:
            rank_one = 0.
        else:
            rank_one = np.outer(zpc, zpc) - np.eye(self.N)
        # Update
        self.Z += (self.cmu * rank_mu + self.cone * rank_one)
        self.t_stall += 1

    def decompose(self):
        if self.t_stall == self.teig:
            # update C
            D = np.linalg.eigvalsh(self.Z)
            fac = min(0.75 / abs(D.min()), 1.)
            self.C = np.dot(np.dot(self.sqrtC, np.eye(self.N) + fac * self.Z), self.sqrtC)

            # force C to be correlation matrix
            cd = np.sqrt(np.diag(self.C))
            self.C = (self.C / cd).T / cd

            # decomposition
            DD, self.B = np.linalg.eigh(self.C)
            self.S = np.sqrt(DD)
            self.sqrtC = np.dot(self.B * self.S, self.B.T)
            self.invsqrtC = np.dot(self.B / self.S, self.B.T)
            self.Z[:, :] = 0.
            self.t_stall = 0

            return cd
        else:
            return 1.


class DdCma:
    
    """dd-CMA: CMA-ES with diagonal decoding

    Original Code
    -------------
    https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518

    Reference
    ---------
    Y. Akimoto and N. Hansen. 
    Diagonal Acceleration for Covariance Matrix Adaptation Evolution Strategies
    Evolutionary Computation (2020) 28(3): 405--435.
    """
    
    def __init__(self, func, xmean0, sigma0, 
                 lam=None,
                 flg_covariance_update=True,
                 flg_variance_update=True,
                 flg_active_update=True,
                 flg_covariance_cold_start=True,
                 flg_tpa=True,
                 beta_thresh=2.,
                 d_thresh=1.5,
                 model_type=VSModel,
                 **kwargs):
        """
        Parameters
        ----------
        func : callable
            parameter : 2d array-like with candidate solutions (x) as elements
            return    : 1d array-like with f(x) as elements
        xmean0 : 1d array-like
            initial mean vector
        sigma0 : 1d array-like
            initial diagonal decoding
        lam : int, optional (default = None)
            population size
        flg_covariance_update : bool, optional (default = True)
            update C if this is True
        flg_variance_update : bool, optional (default = True)
            update D if this is True
        flg_active_update : bool, optional (default = True)
            update C and D with active update
        flg_covariance_cold_start : bool, optional (default = True)
            stall C update at the beginning
        beta_thresh : float, optional (default = 2.)
            threshold parameter for beta control
        d_thresh : float, optional (default = 2.)
            threshold parameter for cold start
        model_type : a derived class of `AbstractModel`
            internal correlation matrix model
        kwargs : options for `model_type`
        """

        self.func = func
        self.N = len(xmean0)
        self.chiN = np.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N * self.N))

        # options
        self.flg_covariance_update = flg_covariance_update
        self.flg_variance_update = flg_variance_update
        self.flg_active_update = flg_active_update
        self.flg_tpa = flg_tpa
        self.beta_thresh = beta_thresh

        # parameters for recombination and step-size adaptation
        self.lam = lam if lam else 4 + int(3 * math.log(self.N)) 
        assert self.lam > 2
        w = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam+1))
        w[w > 0] /= np.sum(np.abs(w[w > 0]))
        w[w < 0] /= np.sum(np.abs(w[w < 0]))
        self.mueff_positive = 1. / np.sum(w[w > 0] ** 2)
        self.mueff_negative = 1. / np.sum(w[w < 0] ** 2)
        self.cm = 1.
        if self.flg_tpa:
            # tpa
            self.cs = 1.0
            self.ds = 4.0
            self.gs = 4.0
            self.ds_factor = 4.0
            self.mva_w = 0.1
        else:
            # csa
            self.cs = (self.mueff_positive + 2.) / (self.N + self.mueff_positive + 5.)
            self.ds = 1. + self.cs + 2. * max(0., math.sqrt((self.mueff_positive - 1.) / (self.N + 1.)) - 1.)
        self.w = w
        self.flg_tpa = flg_tpa

        # parameters for diagonal decoding
        expo = 0.75
        mu_prime = self.mueff_positive + 1. / self.mueff_positive - 2. + self.lam / (2. * self.lam + 10.)
        m = self.N
        self.cdone = 1. / ( 2 * (m / self.N + 1.) * (self.N + 1.) ** expo + self.mueff_positive / 2.)
        self.cdmu = min(1. - self.cdone, mu_prime * self.cdone)
        self.cdc = math.sqrt(self.mueff_positive * self.cdone) / 2.
        self.wd = np.array(w)
        self.wd[w < 0] *= min(1. + self.cdone / self.cdmu, 1. + 2. * self.mueff_negative / (self.mueff_positive + 2.))
        
        # dynamic parameters
        self.xmean = np.array(xmean0)
        self.D = np.array(sigma0)
        self.sigma = 1.
        self.pdc = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        if self.flg_tpa:
            # tpa
            self.dm = np.zeros(self.N)  
            self.mva_dr2 = 0.0
            self.mva_dr1 = 0.0
        else:
            # csa
            self.ps = np.zeros(self.N)

        # internal C model
        self.model = model_type(self.N, self.lam, **kwargs)
        
        # C update stalling
        self.flg_c_update = not flg_covariance_cold_start
        self.d_thresh = d_thresh
        self.moving_weight = 1. / self.N
        self.d_change = np.zeros(self.N)
        self.d_change_abs = np.zeros(self.N)
        self.d_change_mean = 0
        self.d_change_gmean = 0

        # others
        self.neval = 0
        self.t = 0
        self.beta = 1.
        
        # storage for checker and logger
        self.arf = np.zeros(self.lam)
        self.arx = np.zeros((self.lam, self.N))

    def onestep(self):

        # shortcut
        w = self.w
        wc = self.w
        wd = self.wd
            
        # sampling
        arz = np.random.randn(self.lam, self.N)
        ary = self.model.transform(arz) if self.flg_covariance_update else arz
        arx = ary * (self.D * self.sigma) + self.xmean

        # symmetric sampling for tpa
        if self.flg_tpa and self.t > 0: 
            dy = self.dm / self.D
            dz = self.model.inverse_transform(dy)
            znorm = np.linalg.norm(dz)
            nnorm = np.linalg.norm(arz[0])
            #nnorm = np.sqrt(self.N)
            #nnorm = self.chiN
            arz[0] = (nnorm / znorm) * dz
            ary[0] = (nnorm / znorm) * dy
            arx[0] = self.xmean + self.sigma * self.D * ary[0]
            arz[1] = - arz[0]
            ary[1] = - ary[0]
            arx[1] = self.xmean + self.sigma * self.D * ary[1]

        # evaluation
        arf = self.func(arx)
        self.neval += len(arf)
        
        # sort
        idx = np.argsort(arf)
        # if not np.all(arf[idx[1:]] - arf[idx[:-1]] > 0.):
        #     RuntimeWarning("assumed no tie, but there exists")
        sarz = arz[idx]
        sary = ary[idx]
        sarx = arx[idx]

        # recombination
        dz = np.dot(w[w > 0], sarz[w > 0])
        dy = np.dot(w[w > 0], sary[w > 0])
        self.dm = self.cm * self.sigma * self.D * dy
        self.xmean += self.dm
        
        # step-size adaptation        
        self.sigma_old = self.sigma
        if self.flg_tpa:
            rank = np.argsort(idx)
            dr = (rank[1] - rank[0]) / (self.lam - 1)
            hsig = dr < 0.5

            if dr < 0:
                dr /= self.gs
            self.mva_dr1 = (1 - self.mva_w) * self.mva_dr1 + self.mva_w * dr
            self.mva_dr2 = (1 - self.mva_w) * self.mva_dr2 + self.mva_w * dr**2
            var = (3 * (self.gs**2 + 1) * self.lam - (self.gs - 1)**2 * (self.lam + 1)) * (self.lam + 1) / (self.lam - 1)**2 / 36 / self.gs**2
            self.ds = self.ds_factor * abs(self.mva_dr2 - self.mva_dr1**2) / (self.mva_dr1**2 + var / 4)
            self.dr = dr

            # clip
            if dr > 0:
                self.sigma *= math.exp(min(dr / self.ds, 1))
            else:
                self.sigma *= math.exp(max(dr / self.ds, -1))
        else:
            cs = max(1.0 / (self.t + 1), self.cs)
            self.ps = (1 - cs) * self.ps + math.sqrt(cs * (2 - cs) * self.mueff_positive) * dz
            normsquared = np.sum(self.ps * self.ps)
            hsig = normsquared / self.N < 2.0 + 4.0 / (self.N + 1)
            self.sigma *= math.exp((math.sqrt(normsquared) / self.chiN - 1.0) * cs / self.ds)
        
        # cumulaiton
        cc = max(1.0 / (self.t + 1), self.model.cumulation_factor)
        self.pc = (1 - cc) * self.pc + hsig * math.sqrt(cc * (2 - cc) * self.mueff_positive) * self.D * dy
        cdc = max(1.0 / (self.t + 1), self.cdc)
        self.pdc = (1 - cdc) * self.pdc + hsig * math.sqrt(cdc * (2 - cdc) * self.mueff_positive) * self.D * dy 

        # C (intermediate) update
        if self.flg_covariance_update:
            self.model.update(sarz, self.model.inverse_transform(self.pc / self.D))

        # D update
        DD = np.zeros(self.N)
        if self.flg_variance_update:
            DD = self.cdone * (self.model.inverse_transform(self.pdc / self.D) ** 2 - 1.0)
            if self.flg_active_update:
                # positive and negative update
                DD += self.cdmu * np.dot(wd[wd>0], sarz[wd>0] ** 2)
                DD += self.cdmu * np.dot(wd[wd<0] * self.N / np.linalg.norm(sarz[wd<0], axis=1)**2, sarz[wd<0]**2)
                DD -= self.cdmu * np.sum(wd)
            else:
                # positive update
                DD += self.cdmu * np.dot(wd[wd>0], sarz[wd>0] ** 2)
                DD -= self.cdmu * np.sum(wd[wd>0])
            if self.flg_covariance_update:
                self.beta = 1 / max(1, self.model.sqrt_condition_number - self.beta_thresh + 1.)
            else:
                self.beta = 1.
            self.D *= np.exp((self.beta / 2) * DD)

        # update C
        if not self.flg_c_update:
            self.d_change *= (1. - self.moving_weight)
            self.d_change += self.moving_weight * (DD - np.mean(DD)) / math.sqrt(self.cdmu ** 2 * np.sum(wd ** 2) + self.cdone ** 2)
            self.d_change_abs = np.abs(self.d_change) /  math.sqrt(self.moving_weight)
            self.d_change_mean = np.mean(self.d_change_abs)
            self.flg_c_update = np.all(self.d_change_mean  < self.d_thresh) and self.t >= 2. / self.moving_weight
        else:
            if self.flg_covariance_update:
                self.D *= self.model.decompose()
            
        # finalize
        self.t += 1
        self.arf = arf
        self.arx = arx
        
    @property
    def coordinate_std(self):
        if self.flg_covariance_update:
            return self.sigma * self.D * self.model.coordinate_std
        else:
            return self.sigma * self.D

    @property
    def sqrteigvals(self):
        if self.flg_covariance_update:
            return self.model.sqrteigvals
        else:
            return np.ones(self.N)


class Checker:
    """BBOB Termination Checker for dd-CMA"""
    def __init__(self, cma):
        assert isinstance(cma, DdCma)
        self._cma = cma
        self._init_std = self._cma.coordinate_std
        self._N = self._cma.N
        self._lam = self._cma.lam
        self._hist_fbest = deque(maxlen=10 + int(np.ceil(30 * self._N / self._lam)))
        self._hist_feq_flag = deque(maxlen=self._N)
        self._hist_fmin = deque()
        self._hist_fmed = deque()
        
    def __call__(self):
        return self.bbob_check()

    def check_maxiter(self):
        return self._cma.t > 100 + 50 * (self._N + 3) ** 2 / np.sqrt(self._lam)

    def check_tolhistfun(self):
        self._hist_fbest.append(np.min(self._cma.arf))
        return (self._cma.t >= 10 + int(np.ceil(30 * self._N / self._lam)) and
                np.max(self._hist_fbest) - np.min(self._hist_fbest) < 1e-12)

    def check_equalfunvals(self):
        k = int(math.ceil(0.1 + self._lam / 4))
        sarf = np.sort(self._cma.arf)
        self._hist_feq_flag.append(sarf[0] == sarf[k])
        return 3 * sum(self._hist_feq_flag) > self._N

    def check_tolx(self):
        return (np.all(self._cma.coordinate_std / self._init_std) < 1e-12)

    def check_tolupsigma(self):
        return np.any(self._cma.coordinate_std / self._init_std > 1e3)

    def check_stagnation(self):
        self._hist_fmin.append(np.min(self._cma.arf))
        self._hist_fmed.append(np.median(self._cma.arf))
        _len = int(np.ceil(self._cma.t / 5 + 120 + 30 * self._N / self._lam))
        if len(self._hist_fmin) > _len:
            self._hist_fmin.popleft()
            self._hist_fmed.popleft()
        fmin_med = np.median(np.asarray(self._hist_fmin)[-20:])
        fmed_med = np.median(np.asarray(self._hist_fmed)[:20])
        return self._cma.t >= _len and fmin_med >= fmed_med

    def check_conditioncov(self):
        return (self._cma.model.sqrt_condition_number > 1e7
                or np.max(self._cma.D) / np.min(self._cma.D) > 1e7)

    def check_noeffectaxis(self):
        t = self._cma.t % self._N
        test = 0.1 * self._cma.sigma * self._cma.D * self._cma.model.get_axis(t)
        return np.all(self._cma.xmean == self._cma.xmean + test)

    def check_noeffectcoor(self):
        return np.all(self._cma.xmean == self._cma.xmean + 0.2 * self._cma.coordinate_std)

    def check_flat(self):
        return np.max(self._cma.arf) == np.min(self._cma.arf)

    def bbob_check(self):
        if self.check_maxiter():
            return True, 'bbob_maxiter'
        if self.check_tolhistfun():
            return True, 'bbob_tolhistfun'
        if self.check_equalfunvals():
            return True, 'bbob_equalfunvals'
        if self.check_tolx():
            return True, 'bbob_tolx'
        if self.check_tolupsigma():
            return True, 'bbob_tolupsigma'
        if self.check_stagnation():
            return True, 'bbob_stagnation'
        if self.check_conditioncov():
            return True, 'bbob_conditioncov'
        if self.check_noeffectaxis():
            return True, 'bbob_noeffectaxis'
        if self.check_noeffectcoor():
            return True, 'bbob_noeffectcoor'
        if self.check_flat():
            return True, 'bbob_flat'
        return False, ''
    

class Logger:
    """Logger for dd-CMA"""
    def __init__(self, cma, prefix='log', variable_list=['xmean', 'D', 'sqrteigvals', 'sigma', 'beta']):
        """
        Parameters
        ----------
        cma : DdCma instance
        prefix : string
            prefix for the log file path
        variable_list : list of string
            list of names of attributes of `cma` to be monitored
        """
        self.bestsofar = np.inf
        self._cma = cma
        self.prefix = prefix
        self.variable_list = variable_list
        self.logger = dict()
        self.fmin_logger = self.prefix + '_fmin.dat'
        with open(self.fmin_logger, 'w') as f:
            f.write('#' + type(self).__name__ + "\n")
        for key in self.variable_list:
            self.logger[key] = self.prefix + '_' + key + '.dat'
            with open(self.logger[key], 'w') as f:
                f.write('#' + type(self).__name__ + "\n")
                
    def __call__(self, condition=''):
        self.log(condition)

    def log(self, condition=''):
        self.bestsofar = min(np.min(self._cma.arf), self.bestsofar)
        with open(self.fmin_logger, 'a') as f:
            f.write("{} {} {} {}\n".format(self._cma.t, self._cma.neval, np.min(self._cma.arf), self.bestsofar))
            if condition:
                f.write('# End with condition = ' + condition)
        for key, log in self.logger.items():
            key_split = key.split('.')
            key = key_split.pop(0)
            var = getattr(self._cma, key)  
            for i in key_split:
                var = getattr(var, i)  
            if isinstance(var, np.ndarray) and len(var.shape) > 1:
                var = var.flatten()
            varlist = np.hstack((self._cma.t, self._cma.neval, var))
            with open(log, 'a') as f:
                f.write(' '.join(map(repr, varlist)) + "\n")

    def my_formatter(self, x, pos):
        """Float Number Format for Axes"""
        float_str = "{0:2.1e}".format(x)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0}e{1}".format(base, int(exponent))
        else:
            return r"" + float_str + ""
        
    def plot(self,
             xaxis=0,
             ncols=None,
             figsize=None,
             cmap_='Spectral'):
        
        """Plot the result
        Parameters
        ----------
        xaxis : int, optional (default = 0)
            0. vs iterations
            1. vs function evaluations
        ncols : int, optional (default = None)
            number of columns
        figsize : tuple, optional (default = None)
            figure size
        cmap_ : string, optional (default = 'spectral')
            cmap
        
        Returns
        -------
        fig : figure object.
            figure object
        axdict : dictionary of axes
            the keys are the names of variables given in `variable_list`
        """
        mpl.rc('lines', linewidth=2, markersize=8)
        mpl.rc('font', size=12)
        mpl.rc('grid', color='0.75', linestyle=':')
        mpl.rc('ps', useafm=True)  # Force to use
        mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
        mpl.rc('text', usetex=True)  # for a paper submision

        prefix = self.prefix
        variable_list = self.variable_list

        # Default settings
        nfigs = 1 + len(variable_list)
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(nfigs)))
        nrows = int(np.ceil(nfigs / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        axdict = dict()
        
        # Figure
        fig = plt.figure(figsize=figsize)
        # The first figure
        x = np.loadtxt(prefix + '_fmin.dat')
        x = x[~np.isnan(x[:, xaxis]), :]  # remove columns where xaxis is nan
        # Axis
        ax = plt.subplot(nrows, ncols, 1)
        ax.set_title('fmin')
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        plt.plot(x[:, xaxis], x[:, 2:])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        axdict['fmin'] = ax

        # The other figures
        idx = 1
        for key in variable_list:
            idx += 1
            x = np.loadtxt(prefix + '_' + key + '.dat')
            x = x[~np.isnan(
                x[:, xaxis]), :]  # remove columns where xaxis is nan
            ax = plt.subplot(nrows, ncols, idx)
            ax.set_title(r'\detokenize{' + key + '}')
            ax.grid(True)
            ax.grid(which='major', linewidth=0.50)
            ax.grid(which='minor', linewidth=0.25)
            cmap = plt.get_cmap(cmap_)
            cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
            for i in range(x.shape[1] - 2):
                plt.plot(
                    x[:, xaxis], x[:, 2 + i], color=scalarMap.to_rgba(i))
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            axdict[key] = ax

        plt.tight_layout() # NOTE: not sure if it works fine
        return fig, axdict


class BenchmarkFunction:

    def __init__(self, id_func, dim, is_rotated):
        """
        Parameters
        ----------
        id_func : int
            0 : sphere      (x0 = 3. * N(0, 1), sigma0 = 1.) 
            1 : cigar       (x0 = 3. * N(0, 1), sigma0 = 1.) 
            2 : discus      (x0 = 3. * N(0, 1), sigma0 = 1.) 
            3 : ellipsoid   (x0 = 3. * N(0, 1), sigma0 = 1.) 
            4 : twoaxes     (x0 = 3. * N(0, 1), sigma0 = 1.) 
            5 : ellcig      (x0 = 3. * N(0, 1), sigma0 = 1.) 
            6 : ellciglog   (x0 = 3. * N(0, 1), sigma0 = 1.) 
            7 : elldis      (x0 = 3. * N(0, 1), sigma0 = 1.) 
            8 : elldislog   (x0 = 3. * N(0, 1), sigma0 = 1.) 
            9 : cigtab      (x0 = 3. * N(0, 1), sigma0 = 1.) 
           10 : cigtablog   (x0 = 3. * N(0, 1), sigma0 = 1.) 
           11 : ellcigtab   (x0 = 3. * N(0, 1), sigma0 = 1.) 
           12 : ellcigtablog(x0 = 3. * N(0, 1), sigma0 = 1.) 
           13 : rosenbrock  (x0 = .1 * N(0, 1), sigma0 = .1) 
           14 : bohachevsky (x0 = 8. * N(0, 1)., sigma0 = 7.) 
           15 : rastrigin   (x0 = 3. * N(0, 1)., sigma0 = 2.) 

        is_rotated : bool
        """
        if id_func == 0:
            self.func = self.sphere
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 1:
            R = random_axes(dim, 1) if is_rotated else np.eye(1, dim)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-3]), coordinate_scales=1e3)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 2:
            R = random_axes(dim, 1) if is_rotated else np.eye(1, dim)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e3]), coordinate_scales=1e0)                
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 3:
            func = self.ellipsoid
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 4:
            func = self.twoaxes
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 5:
            R = random_axes(dim, 1)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2]), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 6:
            n_axes = int(math.ceil(math.log(dim)))
            R = random_axes(dim, n_axes)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2] * n_axes), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 7:
            R = random_axes(dim, 1)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e2]), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 8:
            n_axes = int(math.ceil(math.log(dim)))
            R = random_axes(dim, n_axes)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e2] * n_axes), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 9:
            R = random_axes(dim, 2)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2, 1e2]), coordinate_scales=1.0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 10:
            n_axes = int(math.ceil(math.log(dim)))
            R = random_axes(dim, n_axes * 2)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2] * n_axes + [1e2] * n_axes), coordinate_scales=1.0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 11:
            R = random_axes(dim, 2)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2, 1e2]), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 12:
            n_axes = int(math.ceil(math.log(dim)))
            R = random_axes(dim, n_axes * 2)
            func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-2] * n_axes + [1e2] * n_axes), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 13:
            func = self.rosenbrock
            self.func = self.random_rotation(func, dim) if is_rotated else func            
            self.x0 = 0.1 * np.random.randn(dim)
            self.sigma0 = 0.1 * np.ones(dim)
        else:
            raise NotImplementedError

    def __call__(self, x):
        return self.func(x)

    def get_initial_distribution(self):
        return self.x0, self.sigma0

    def random_rotation(self, func, dim):
        R = random_axes(dim, dim)
        def rotatedfunc(x):
            return func(np.dot(x, R.T))
        return rotatedfunc

    def sphere(self, x):
        return np.sum(x ** 2, axis=1)

    def ellipsoid(self, x):
        a = 1e3
        dim = x.shape[1]
        diaghess = a**(2.0 * np.arange(dim) / float(dim - 1))
        return np.dot(x ** 2, diaghess)

    def twoaxes(self, x):
        a = 1e3
        dim = x.shape[1]
        diaghess = np.array([1e6] * (dim//2) + [1.0] * (dim - dim//2))
        return np.dot(x ** 2, diaghess)

    def ellcigtab(self, x, axes, axis_scales, coordinate_scales):
        y = x * coordinate_scales
        fx = np.sum(y ** 2, axis=1)
        fx += np.dot(np.dot(y, axes.T) ** 2, axis_scales**2 - 1.) 
        return fx
    
    def rosenbrock(self, x):
        a = 1e2                
        return a * np.sum(
            (x[:, :-1]**2 - x[:, 1:])**2, axis=1) + np.sum(
                (x[:, :-1] - 1.0)**2, axis=1)
    

if __name__ == "__main__":
    """USAGE: python ddrcma.py 20 0 0 'range(13)' 'range(1)' 1
    argv[1] : dimension
    argv[2] : 0: original coordinate, 1: rotated coordinate
    argv[3] : 0: dd-vs-cma-tpa, 1: dd-vs-cma-csa, 2: dd-cma-csa
    argv[4] : function id list (string of iterable)
    argv[5] : seed list (string of iterable)
    argv[6] : 0: no figure is produced, 1: figure is produced
    """
    import sys
    import numpy as np

    N = int(sys.argv[1])
    rot = int(sys.argv[2])
    mode = int(sys.argv[3])
    funclist = eval(sys.argv[4])
    seedlist = eval(sys.argv[5])
    figure = int(sys.argv[6])


    for seed in seedlist:
        np.random.seed(20190505 * (seed + 1))

        for functype in funclist:
            func = BenchmarkFunction(functype, N, rot)
            xmean0, sigma0 = func.get_initial_distribution()

            ftarget = 1e-8
            maxitr = int(1e4 * N)

            # Main loop
            if mode == 0:
                ddcma = DdCma(func=func, xmean0=xmean0, sigma0=sigma0, model_type=VSModel, flg_tpa=True, beta_cond=3e1)
                logger = Logger(ddcma, prefix="../dat/ddrcmatpa_func{}_dim{}_rot{}_seed{}".format(functype, N, rot, seed), variable_list=['xmean', 'D', 'sqrteigvals', 'sigma', 'beta', 'model.kshort', 'model.klong'])
            elif mode == 1:
                ddcma = DdCma(func=func, xmean0=xmean0, sigma0=sigma0, model_type=VSModel, flg_tpa=False, beta_cond=3e1)
                logger = Logger(ddcma, prefix="../dat/ddrcmacsa_func{}_dim{}_rot{}_seed{}".format(functype, N, rot, seed), variable_list=['xmean', 'D', 'sqrteigvals', 'sigma', 'beta', 'model.kshort', 'model.klong'])
            elif mode == 2:
                ddcma = DdCma(func=func, xmean0=xmean0, sigma0=sigma0, model_type=FullModel, flg_covariance_cold_start=False)
                logger = Logger(ddcma, prefix="../dat/ddcmacsa_func{}_dim{}_rot{}_seed{}".format(functype, N, rot, seed), variable_list=['xmean', 'D', 'sqrteigvals', 'sigma', 'beta'])
            else:
                raise ValueError
            checker = Checker(ddcma)
            issatisfied = False
            fbestsofar = np.inf
            for itr in range(maxitr):
                ddcma.onestep()
                fbest = np.min(ddcma.arf)
                fbestsofar = min(fbest, fbestsofar)
                if fbest < ftarget:
                    issatisfied, condition = True, 'ftarget'
                else:
                    issatisfied, condition = checker()
                if ddcma.t % (N//10) == 0:
                    print(ddcma.t, ddcma.neval, fbest, fbestsofar)
                    logger()
                if issatisfied:
                    break
            print(ddcma.t, ddcma.neval, fbest, fbestsofar)
            print("Terminated with condition: " + condition)
            logger(condition)

            if figure:
                # Produce a figure
                fig, axdict = logger.plot()
                for key in axdict:
                    if key not in ('xmean'):
                        axdict[key].set_yscale('log')
                plt.savefig(logger.prefix + '.pdf')
