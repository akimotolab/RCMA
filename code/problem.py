from functools import partial
import math
import numpy as np


def random_axes(dim, n_axes):
    R = np.random.normal(0, 1, (n_axes, dim))
    for i in range(n_axes):
        for j in range(i):
            R[i] = R[i] - np.dot(R[i], R[j]) * R[j]
        R[i] = R[i] / np.linalg.norm(R[i])
    return R


class BenchmarkFunction:

    def __init__(self, id_func, dim):

        if id_func == 0:
            # sphere
            self.func = self.sphere
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 1:
            # cigar
            k = int(math.log(dim))
            R = np.eye(k, dim)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-3]*k), coordinate_scales=1e0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 2:
            # ellipsoid
            self.func = self.ellipsoid
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 3:
            # discus
            k = int(math.log(dim))
            R = np.eye(k, dim)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e3]*k), coordinate_scales=1e0)                
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 4:
            # two axes
            self.func = self.twoaxes
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 5:
            # cigar discus
            k = int(math.log(dim))
            R = np.eye(k * 2, dim)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([10**(-1.5)]*k + [10**(1.5)]*k), coordinate_scales=1.0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 6:
            # rotated cigar
            k = int(math.log(dim))
            R = random_axes(dim, k)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e-3]*k), coordinate_scales=1e0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 7:
            # rotated two axes
            self.func = self.random_rotation(self.twoaxes, dim)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 8:
            # rotated discus
            k = int(math.log(dim))
            R = random_axes(dim, k) 
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([1e3]*k), coordinate_scales=1e0)                
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 9:
            # rotated cigar discus
            k = int(math.log(dim))
            R = random_axes(dim, k * 2)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([10**(-2)]*k + [10**(2)]*k), coordinate_scales=1.0)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 10:
            # ellipsoid cigar
            k = int(math.log(dim))
            R = random_axes(dim, k)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([10**(-2)]*k), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 11:
            # ellipsoid discus
            k = int(math.log(dim))
            R = random_axes(dim, k)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([10**(2)]*k), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 12:
            # ellipsoid cigar discus
            k = int(math.log(dim))
            R = random_axes(dim, k * 2)
            self.func = partial(self.ellcigtab, axes=R, axis_scales=np.array([10**(-1)]*k + [10**(1)]*k), coordinate_scales=np.logspace(0, 1, dim, base=1e2))
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 13:
            # subspace roatated ellipsoid
            k = int(2 * math.log(dim))
            self.func = self.subspace_random_rotation(self.ellipsoid, dim, k)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 14:
            # 4-blocks rotated ellipsoid
            self.func = self.fourblock_random_rotation(self.ellipsoid, dim)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 15:
            # rotated ellipsoid
            self.func = self.random_rotation(self.ellipsoid, dim)
            self.x0 = 3.0 * np.random.randn(dim)
            self.sigma0 = 1.0 * np.ones(dim)
        elif id_func == 16:
            # rosenbrock
            self.func = self.rosenbrock
            self.x0 = 0.1 * np.random.randn(dim)
            self.sigma0 = 0.1 * np.ones(dim)
        elif id_func == 17:
            # rotated rosenbrock
            func = self.rosenbrock
            self.func = self.random_rotation(func, dim)
            self.x0 = 0.1 * np.random.randn(dim)
            self.sigma0 = 0.1 * np.ones(dim)
        else:
            raise NotImplementedError

    def __call__(self, x):
        return self.func(x)

    def get_initial_distribution(self):
        return self.x0, self.sigma0

    def get_initial_fvalue(self):
        return self(self.x0.reshape((1, -1)))[0]

    def random_rotation(self, func, dim):
        R = random_axes(dim, dim)
        def rotatedfunc(x):
            return func(np.dot(x, R.T))
        return rotatedfunc

    def subspace_random_rotation(self, func, dim, subdim):
        R = random_axes(dim, subdim)
        def rotatedfunc(x):
            return func(np.dot(x, R.T))
        return rotatedfunc

    def fourblock_random_rotation(self, func, dim):
        dim1 = dim // 4
        dim2 = dim - dim1 * 3
        R1 = random_axes(dim1, dim1)
        R2 = random_axes(dim1, dim1)
        R3 = random_axes(dim1, dim1)
        R4 = random_axes(dim2, dim2)
        def rotatedfunc(x):
            y = np.zeros(x.shape)
            y[:, :dim1] = np.dot(x[:, :dim1], R1.T)
            y[:, dim1:2*dim1] = np.dot(x[:, dim1:2*dim1], R2.T)
            y[:, 2*dim1:3*dim1] = np.dot(x[:, 2*dim1:3*dim1], R3.T)
            y[:, -dim2:] = np.dot(x[:, -dim2:], R4.T)
            return func(y)
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
    