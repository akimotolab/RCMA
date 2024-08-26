import math
from time import perf_counter
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('lines', linewidth=2, markersize=8)
mpl.rc('font', size=12)
mpl.rc('grid', color='0.75', linestyle=':')
mpl.rc('ps', useafm=True)  # Force to use
mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
mpl.rc('text', usetex=True)  # for a paper submision
mpl.rc('figure', autolayout=True)

if __name__ == "__main__":
    
    str2list = lambda x:list(map(int, x.split(',')))
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    parser.add_argument('-m', '--method', type=str2list, default="0,1,2,3,4,5,6,11", help='0:dd-cma, 1:cma, 2:sep-cma, 3:dd-rcma, 4:dd-rcma-full, 5:vkd-cma, 6:vkd-cma-noadapt, 7:lmma')
    parser.add_argument('-s', '--seed', type=str2list, default="100,200,300,400,500", help="seed list")
    parser.add_argument('-f', '--function', type=str2list, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", help="function ID list, separated by comma")
    parser.add_argument('-d', '--dimension', type=str2list, default="80,160,320,640,1280,2560,5120,10240", help="dimension list, separated by comma")
    parser.add_argument('-t', '--target', type=float, default=1e-10, help="target threshold")
    parser.add_argument('--datapath', type=str, default='../dat/', help="path to the output directory")
    parser.add_argument('--figpath', type=str, default='../fig/', help="path to the output directory")

    args = parser.parse_args()

    LS = [":", ":", ":", "-", "-", "--", "--", "-."]    
    MS = ["x", "o", "+", "<", ">", "v", "^", "*"]    
    COLOR = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # LS = [":x", ":o", ":+", "->", "--v", "--^", "-.*"]    

    for funcid in args.function:
        plt.figure(figsize=(3,3))
        for imethod, method in enumerate(args.method):
            dat = np.zeros((len(args.dimension), len(args.seed)))
            for iseed, seed in enumerate(args.seed):
                for idim, dim in enumerate(args.dimension):
                    if dim >= 1280 and (method <= 1 or funcid in (7,15,16,17)):
                        dat[idim, iseed] = np.nan
                        continue
                    filename = args.datapath + "method{}_func{}_dim{}_seed{}.txt".format(method, funcid, dim, seed)
                    try:
                        print(filename)
                        hist = np.loadtxt(filename)
                        print("file loaded")
                        idxlist = np.nonzero(hist[:, 2] <= args.target)[0]
                        if (not (funcid in (7,15,16,17))) and hist[-1, 0] > 1e4 * dim:
                            dat[idim, iseed] = np.nan
                        elif (funcid in (7,15,16,17)) and hist[-1, 0] > 1e5 * dim:
                            dat[idim, iseed] = np.nan
                        elif len(idxlist) > 0:
                            dat[idim, iseed] = hist[idxlist[0], 0] / dim
                        else:
                            dat[idim, iseed] = np.nan
                    except:
                        dat[idim, iseed] = np.nan
            dat = np.sort(dat, axis=1)
            plt.loglog(args.dimension, dat[:, :], MS[imethod], fillstyle='none', color=COLOR[imethod], alpha=0.8)
            plt.loglog(args.dimension, dat[:, 2], LS[imethod], fillstyle='none', color=COLOR[imethod], alpha=0.8)
        # plt.ylabel('no. func. evals. / dim.')
        # plt.xlabel('dim.')
        plt.xlim((np.min(args.dimension)/2, np.max(args.dimension)*2))
        if funcid in (7,15,16,17):
            plt.ylim((1e2, 1e5))
            plt.xlim((80/2, 640*2))
        else:
            plt.ylim((1e1, 1e4))
            plt.xlim((np.min(args.dimension)/2, np.max(args.dimension)*2))
        plt.grid()
        plt.tight_layout()
        plt.savefig(args.figpath + 'func{}.pdf'.format(funcid))