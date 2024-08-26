import math
from time import process_time
import traceback
import argparse
from collections import deque
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ddcma
import ddrcma
import lmmaes
import vkdcma
import vdcma
import problem
import pylmcma

class Logger:
    def __init__(self, cma, prefix='log', variable_list=['xmean', 'D', 'sqrteigvals', 'sigma', 'beta']):
        """
        Parameters
        ----------
        cma : Optimizer instance
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
                
    def __call__(self, itr, neval, condition=''):
        self.log(itr, neval, condition)

    def log(self, itr, neval, condition=''):
        self.bestsofar = min(self._cma.fbest(), self.bestsofar)
        with open(self.fmin_logger, 'a') as f:
            f.write("{} {} {} {}\n".format(itr, neval, self._cma.fbest(), self.bestsofar))
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
            varlist = np.hstack((itr, neval, var))
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


class Checker:
    """BBOB Termination Checker for Optimizer"""
    def __init__(self, cma, N, lam):
        assert isinstance(cma, Optimizer)
        self._cma = cma
        self._init_std = self._cma.coordinate_std
        self._N = N
        self._lam = lam
        self._hist_fbest = deque(maxlen=10 + int(np.ceil(30 * self._N / self._lam)))
        self._hist_feq_flag = deque(maxlen=self._N)
        self._hist_fmin = deque()
        self._hist_fmed = deque()
        
    def __call__(self, t):
        return self.bbob_check(t)

    def check_tolhistfun(self, t):
        self._hist_fbest.append(np.min(self._cma.flist))
        return (t >= 10 + int(np.ceil(30 * self._N / self._lam)) and
                np.max(self._hist_fbest) - np.min(self._hist_fbest) < 1e-12)

    def check_equalfunvals(self):
        k = int(math.ceil(0.1 + self._lam / 4))
        sarf = np.sort(self._cma.flist)
        self._hist_feq_flag.append(sarf[0] == sarf[k])
        return 3 * sum(self._hist_feq_flag) > self._N

    def check_tolx(self):
        return (np.all(self._cma.coordinate_std / self._init_std) < 1e-12)

    def check_tolupsigma(self):
        return np.any(self._cma.coordinate_std / self._init_std > 1e3)

    def check_stagnation(self, t):
        self._hist_fmin.append(np.min(self._cma.flist))
        self._hist_fmed.append(np.median(self._cma.flist))
        _len = int(np.ceil(t / 5 + 120 + 30 * self._N / self._lam))
        if len(self._hist_fmin) > _len:
            self._hist_fmin.popleft()
            self._hist_fmed.popleft()
        fmin_med = np.median(np.asarray(self._hist_fmin)[-20:])
        fmed_med = np.median(np.asarray(self._hist_fmed)[:20])
        return t >= _len and fmin_med >= fmed_med

    def check_noeffectcoor(self):
        return np.all(self._cma.xmean == self._cma.xmean + 0.2 * self._cma.coordinate_std)


    def bbob_check(self, t):
        if self.check_tolhistfun(t):
            return True, 'bbob_tolhistfun'
        if self.check_equalfunvals():
            return True, 'bbob_equalfunvals'
        if self.check_tolx():
            return True, 'bbob_tolx'
        if self.check_tolupsigma():
            return True, 'bbob_tolupsigma'
        # if self.check_stagnation(t):
        #     return True, 'bbob_stagnation'
        if self.check_noeffectcoor():
            return True, 'bbob_noeffectcoor'
        return False, ''

class Optimizer:
    def onestep(self):
        raise NotImplementedError
    def fbest(self):
        raise NotImplementedError
    @property
    def flist(self):
        raise NotImplementedError
    @property
    def coordinate_std(self):
        raise NotImplementedError
    @property
    def xmean(self):
        raise NotImplementedError


class LMCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        self.dim = len(xmean0)
        self.lam = 4 + int(3 * math.log(self.dim)) 
        seed = np.random.randint(0, 10000)
        self.es = pylmcma.LMCMA(self.dim, self.lam, seed, sigma0[0], xmean0)
        self.func = func
        self.arf = np.zeros(self.lam)
    def onestep(self):
        arx = self.es.getarx()
        self.arf = self.func(arx)
        self.es.setarf(self.arf)
        self.es.update()
    def fbest(self):
        return self.arf.min()
    def stepsize(self):
        return 1.0
    @property
    def flist(self):
        return self.arf
    @property
    def coordinate_std(self):
        return np.ones(self.dim)
    @property
    def xmean(self):
        return np.ones(self.dim)


class DDCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        self.es = ddcma.DdCma(xmean0, sigma0)
        self.func = func
    def onestep(self):
        self.es.onestep(self.func)
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean


class DDCMATPA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        self.es = ddrcma.DdCma(func, xmean0, sigma0, model_type=ddrcma.FullModel, flg_covariance_cold_start=False)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean

class CMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        self.es = ddcma.DdCma(xmean0, sigma0, flg_variance_update=False)
        self.func = func
    def onestep(self):
        self.es.onestep(self.func)
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean

class SEPCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        self.es = ddcma.DdCma(xmean0, sigma0, flg_covariance_update=False)
        self.func = func
    def onestep(self):
        self.es.onestep(self.func)
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean


class DDRCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        lam = 4 + int(3 * math.log(dim)) 
        self.es = ddrcma.DdCma(func, xmean0, sigma0, model_type=ddrcma.VSModel, kmax=lam)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean

class DDRCMAFULL(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        self.es = ddrcma.DdCma(func, xmean0, sigma0, model_type=ddrcma.VSModel, kmax=dim-1)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean

class VKDCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        lam = 4 + int(3 * math.log(dim)) 
        self.es = vkdcma.VkdCma(func, xmean0, sigma0, kmax=lam, batch_evaluation=True)
    def onestep(self):
        self.es._onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        ka = self.es.k_active
        if ka == 0:
            return self.es.sigma * self.es.D
        else:
            int_coordinate_std = np.sqrt(1.0 + np.dot(self.es.S[:ka], self.es.V[:ka] ** 2))
            return self.es.sigma * self.es.D * int_coordinate_std
    @property
    def xmean(self):
        return self.es.xmean


class VKDCMANOADAPT(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        lam = 4 + int(3 * math.log(dim)) 
        self.es = vkdcma.VkdCma(func, xmean0, sigma0, k_init=lam, kmin=lam, kmax=lam, batch_evaluation=True)
    def onestep(self):
        self.es._onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        ka = self.es.k_active
        if ka == 0:
            return self.es.sigma * self.es.D
        else:
            int_coordinate_std = np.sqrt(1.0 + np.dot(self.es.S[:ka], self.es.V[:ka] ** 2))
            return self.es.sigma * self.es.D * int_coordinate_std
    @property
    def xmean(self):
        return self.es.xmean


class LMMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        self.es = lmmaes.LmMa(func, xmean0, sigma0)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma[0]
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.sigma
    @property
    def xmean(self):
        return self.es.xmean


class RCMA(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        lam = 4 + int(3 * math.log(dim)) 
        self.es = ddrcma.DdCma(func, xmean0, sigma0, model_type=ddrcma.VSModel, kmax=lam, flg_variance_update=False)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma
    @property
    def flist(self):
        return self.es.arf
    @property
    def coordinate_std(self):
        return self.es.coordinate_std
    @property
    def xmean(self):
        return self.es.xmean

class RCMAFULL(Optimizer):
    def __init__(self, func, xmean0, sigma0):
        dim = len(xmean0)
        self.es = ddrcma.DdCma(func, xmean0, sigma0, model_type=ddrcma.VSModel, kmax=dim-1, flg_variance_update=False)
    def onestep(self):
        self.es.onestep()
    def fbest(self):
        return self.es.arf.min()
    def stepsize(self):
        return self.es.sigma

OptimizerList = [DDCMA, CMA, SEPCMA, DDRCMA, DDRCMAFULL, VKDCMA, VKDCMANOADAPT, LMMA, RCMA, RCMAFULL, DDCMATPA, LMCMA]

if __name__ == "__main__":
    
    str2list = lambda x:list(map(int, x.split(',')))
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    parser.add_argument('-m', '--method', type=int, required=True, help='0:dd-cma, 1:cma, 2:sep-cma, 3:dd-rcma, 4:dd-rcma-full, 5:vkd-cma, 6:vkd-cma-noadapt, 7:lmma, 8:rcma, 9:rcma-full, 10:ddcma-tpa, 11:lm-cma')
    parser.add_argument('-s', '--seed', type=int, required=True, help="seed")
    parser.add_argument('-f', '--function', type=str2list, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", help="function ID list, separated by comma")
    parser.add_argument('-d', '--dimension', type=str2list, default="80,160,320,640,1280,2560,5120,10240", help="dimension list, separated by comma")
    parser.add_argument('--maxeval', type=int, default=10000, help="budget (f-calls) = maxeval * dimension")
    parser.add_argument('--maxsec', type=float, default=float(60*60*24*365), help="budget (cpu-time) = maxsec")
    parser.add_argument('--target', type=float, default=1e-10, help="budget (cpu-time) = maxsec")
    parser.add_argument('--path', type=str, default='../dat/', help="path to the output directory")
    parser.add_argument('--debug', type=bool, default=False, help="debug mode")

    args = parser.parse_args()

    for dim in args.dimension:
        for funcid in args.function:
            # seed
            np.random.seed(args.seed)
            # problem
            func = problem.BenchmarkFunction(funcid, dim)
            xmean0, sigma0 = func.get_initial_distribution()
            f0 = func.get_initial_fvalue()
            ftarget = args.target
            maxeval = args.maxeval * dim
            maxsec = args.maxsec
            lam = 4 + int(3 * math.log(dim))
            maxitr = maxeval // lam + 1 
            # method 
            elapsed_time = 0.0
            elapsed_eval = 0
            fbest = 1.0
            method = OptimizerList[args.method](func, xmean0, sigma0)
            sigma = method.stepsize()
            # termination
            checker = Checker(method, dim, lam)
            # logger
            filename = args.path + "method{}_func{}_dim{}_seed{}.txt".format(args.method, funcid, dim, args.seed)
            with open(filename, "w") as f:
                f.write("{} {} {} {}\n".format(elapsed_eval, elapsed_time, fbest, sigma))

            # debug
            if args.debug:
                if isinstance(method, (DDRCMA, DDRCMAFULL, RCMA)):
                    logger = Logger(method, prefix=args.path + "debug", variable_list=['es.xmean', 'es.D', 'es.sqrteigvals', 'es.sigma', 'es.beta', 'es.model.klong', 'es.model.kshort'])
                if isinstance(method, DDCMA):
                    logger = Logger(method, prefix=args.path + "debug", variable_list=['es.xmean', 'es.D', 'es.S', 'es.sigma', 'es.beta'])
                if isinstance(method, DDCMATPA):
                    logger = Logger(method, prefix=args.path + "debug", variable_list=['es.xmean', 'es.D', 'es.sqrteigvals', 'es.sigma'])

            try:
                for t in range(maxitr):
                    ts = process_time()
                    method.onestep()
                    te = process_time()
                    elapsed_time += te - ts
                    elapsed_eval += lam
                    fbest = method.fbest()
                    sigma = method.stepsize()
                    # termination check
                    if fbest / f0 <= ftarget:
                        condition = "ftarget"
                        break
                    if elapsed_eval >= maxeval:
                        condition = "maxeval"
                        break
                    if elapsed_time >= maxsec:
                        condition = "maxsec"
                        break
                    flg, condition = checker(t+1)
                    if flg:
                        break
                    # log
                    if t % (dim//10) == 0:
                        with open(filename, "a") as f:
                            f.write("{} {} {} {}\n".format(elapsed_eval, elapsed_time, fbest / f0, sigma))
                        if args.debug:
                            logger(t+1, elapsed_eval)
                # final log 
                with open(filename, "a") as f:
                    f.write("{} {} {} {}\n".format(elapsed_eval, elapsed_time, fbest / f0, sigma))
                    f.write("#" + condition)
                if args.debug:
                    logger(t+1, elapsed_eval)

                # if isinstance(method, (DDRCMA, DDRCMAFULL)):
                #     covfilename = args.path + "method{}_func{}_dim{}_seed{}_chol.txt".format(args.method, funcid, dim, args.seed)
                #     chol = method.es.model.transform(np.eye(dim)) * (method.es.D * method.es.sigma)
                #     np.savetxt(covfilename, chol)
                #     meanfilename = args.path + "method{}_func{}_dim{}_seed{}_mean.txt".format(args.method, funcid, dim, args.seed)
                #     xmean = method.es.xmean
                #     np.savetxt(meanfilename, xmean)

            except Exception as e:
                with open(filename, "a") as f:
                    f.write("#exception")
                print("M{} D{} F{} S{}".format(args.method, dim, funcid, args.seed))
                print("{} {} {} {}\n".format(elapsed_eval, elapsed_time, fbest / f0, sigma))
                print(traceback.format_exc())

            if args.debug:
                fig, axdict = logger.plot()
                for key in axdict:
                    if key not in ('xmean'):
                        axdict[key].set_yscale('log')
                plt.savefig(logger.prefix + '.pdf')