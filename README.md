# RCMA
dd-RCMA-ES : Restricted Covariance Matrix Adaptation Evolution Strategy for High Dimensional Problems

## Code

* ddrcma.py : DD-RCMA-ES code
* lmmaes.py : LM-MA-ES code
* pylmcma.py : LM-CMA-ES wrapper. The original C++ implementation is downloaded from [Author's Page](http://loshchilov.com/index.html)
* ddcma.py : DD-CMA-ES code downloaded on 2022/03/25 from [GitHub Gist](https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518)
* vkdcma.py : VkD-CMA-ES code downloaded on 2022/03/25 from [GitHub Gist](https://gist.github.com/youheiakimoto/2fb26c0ace43c22b8f19c7796e69e108)
* vdcma.py : VD-CMA-ES code downloaded on 2022/03/25 from [GitHub Gist](https://gist.github.com/youheiakimoto/08b95b52dfbf8832afc71dfff3aed6c8)
* problem.py : problem definitions
* benchmarking.py : benchmarking script
* plot.py : plot script

## Compile (for LM-CMA-ES)

```
g++ -fPIC -c pylmcma.cpp
g++ -fPIC -shared -o pylmcmaclib.so pylmcmaclib.cpp pylmcma.o
```
