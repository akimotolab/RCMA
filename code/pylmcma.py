import ctypes
import numpy as np
import os

# g++ -fPIC -c pylmcma.cpp
# g++ -fPIC -shared -o pylmcmaclib.so pylmcmaclib.cpp pylmcma.o


LMCMAHandle = ctypes.POINTER(ctypes.c_char)

class LMCMA:
    def __init__(self, N, lam, inseed, sigma, xmean, clibname="pylmcmaclib.so", clibpath=os.path.dirname(os.path.abspath(__file__))):
        self._lib = np.ctypeslib.load_library(clibname, clibpath)

        # declare the argument types
        self._lib.createMyLMCMA.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64)
        ]
        # declare the output types
        self._lib.createMyLMCMA.restype = LMCMAHandle

        # declare the argument types
        self._lib.deleteMyLMCMA.argtypes = [ LMCMAHandle ]
        # declare the output types
        self._lib.deleteMyLMCMA.restype = None

        # declare the argument types
        self._lib.getarx.argtypes = [ LMCMAHandle, np.ctypeslib.ndpointer(dtype=np.float64, shape=(lam, N)) ]
        # declare the output types
        self._lib.getarx.restype = None

        # declare the argument types
        self._lib.setarf.argtypes = [ LMCMAHandle, np.ctypeslib.ndpointer(dtype=np.float64, shape=(lam,)) ]
        # declare the output types
        self._lib.setarf.restype = None

        # declare the argument types
        self._lib.update.argtypes = [ LMCMAHandle ]
        # declare the output types
        self._lib.setarf.restype = None

        c_N = ctypes.c_int(N)
        c_lam = ctypes.c_int(lam)
        c_inseed = ctypes.c_int(inseed)
        c_sigma = ctypes.c_double(sigma)

        self._lmcma = self._lib.createMyLMCMA(c_N, c_lam, c_inseed, c_sigma, xmean)
        self.N = N
        self.lam = lam
        self._arx = np.zeros((lam, N), dtype=np.float64)

    def __del__(self):
        self._lib.deleteMyLMCMA(self._lmcma)

    def getarx(self):
        self._lib.getarx(self._lmcma, self._arx)
        return self._arx
    
    def setarf(self, arf):
        return self._lib.setarf(self._lmcma, arf)

    def update(self):
        self._lib.update(self._lmcma)


if __name__ == '__main__':

    lmcma = LMCMA(N=10, lam=3, inseed=100, sigma=1, xmean=np.ones(10), clibname="pylmcmaclib.so", clibpath=".")
    print(lmcma.getarx())