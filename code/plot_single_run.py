import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def my_formatter(x, pos):
    """Float Number Format for Axes"""
    float_str = "{0:2.1e}".format(x)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}e{1}".format(base, int(exponent))
    else:
        return r"" + float_str + ""


def plot_ddcma(prefix, xlim=None, y1lim=None, y2lim=None, y3lim=None, y4lim=None, nsample=300):
    cmap_='Spectral'

    # Default settings
    nfigs = 4
    ncols = 4
    nrows = 1
    figsize = (3 * ncols, 3 * nrows)
    axdict = dict()
    
    # Figure
    fig = plt.figure(figsize=figsize)
    # The first figure
    x = np.loadtxt(prefix + '_fmin.dat')
    x = x[~np.isnan(x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling

    x2 = np.loadtxt(prefix + '_es.sigma.dat')
    x2 = x2[~np.isnan(
        x2[:, 0]), :]  # remove columns where xaxis is nan
    t2 = np.linspace(0, len(x2)-1, nsample, dtype=int) # subsampling

    # Axis
    ax = plt.subplot(nrows, ncols, 1)
    ax.set_title(r'$f(x_{1:\lambda})$, $\sigma$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    plt.plot(x[t, 0], x[t, 2])
    plt.plot(x2[t2, 0], x2[t2, 2])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y1lim is not None:
        ax.set_ylim(y1lim)

    # xmean
    x = np.loadtxt(prefix + '_es.xmean.dat')
    x = x[~np.isnan(x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling    
    ax = plt.subplot(nrows, ncols, 2)
    ax.set_title(r'$m$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    if xlim is not None:
        ax.set_xlim(xlim)
    if y2lim is not None:
        ax.set_ylim(y2lim)

    # D
    x = np.loadtxt(prefix + '_es.D.dat')
    x = x[~np.isnan(
        x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling
    ax = plt.subplot(nrows, ncols, 3)
    ax.set_title(r'$D$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(
            x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y3lim is not None:
        ax.set_ylim(y3lim)

    # sqrt eig
    x = np.loadtxt(prefix + '_es.S.dat')
    x = x[~np.isnan(
        x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling
    ax = plt.subplot(nrows, ncols, 4)
    ax.set_title(r'eig. of $\sqrt{C}$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(
            x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y4lim is not None:
        ax.set_ylim(y4lim)

    plt.tight_layout() # NOTE: not sure if it works fine
    plt.savefig(prefix + '.pdf')
    plt.close()


def plot_ddrcma(prefix, xlim=None, y1lim=None, y2lim=None, y3lim=None, y4lim=None, nsample=300, n_xticks=4):
    cmap_='Spectral'

    # Default settings
    nfigs = 4
    ncols = 4
    nrows = 1
    figsize = (3 * ncols, 3 * nrows)
    axdict = dict()
    
    # Figure
    fig = plt.figure(figsize=figsize)
    # The first figure
    x = np.loadtxt(prefix + '_fmin.dat')
    x = x[~np.isnan(x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling

    x2 = np.loadtxt(prefix + '_es.sigma.dat')
    x2 = x2[~np.isnan(x2[:, 0]), :]  # remove columns where xaxis is nan
    t2 = np.linspace(0, len(x2)-1, nsample, dtype=int) # subsampling

    # Axis
    ax = plt.subplot(nrows, ncols, 1)
    ax.set_title(r'$f(x_{1:\lambda})$, $\sigma$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    plt.plot(x[t, 0], x[t, 2])
    plt.plot(x2[t2, 0], x2[t2, 2])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y1lim is not None:
        ax.set_ylim(y1lim)
    xmax = ax.get_xlim()[1]
    xstep = xmax / n_xticks
    xmagnitude = 10 ** (len(str(int(xstep))) - 1)
    xstep = round(xstep / xmagnitude) * xmagnitude
    ticks = np.arange(0, xmax + 1, xstep)
    ax.set_xticks(ticks)

    # xmean
    x = np.loadtxt(prefix + '_es.xmean.dat')
    x = x[~np.isnan(x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling
    ax = plt.subplot(nrows, ncols, 2)
    ax.set_title(r'$m$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    if xlim is not None:
        ax.set_xlim(xlim)
    if y2lim is not None:
        ax.set_ylim(y2lim)
    ax.set_xticks(ticks)

    # D
    x = np.loadtxt(prefix + '_es.D.dat')
    x = x[~np.isnan(
        x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling
    ax = plt.subplot(nrows, ncols, 3)
    ax.set_title(r'$D$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(
            x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y3lim is not None:
        ax.set_ylim(y3lim)
    ax.set_xticks(ticks)

    # sqrt eig
    x = np.loadtxt(prefix + '_es.sqrteigvals.dat')
    x = x[~np.isnan(
        x[:, 0]), :]  # remove columns where xaxis is nan
    t = np.linspace(0, len(x)-1, nsample, dtype=int) # subsampling
    ax = plt.subplot(nrows, ncols, 4)
    ax.set_title(r'eig. of $\sqrt{C}$')
    ax.grid(True, which='major', linewidth=0.50)
    ax.grid(True, which='minor', linewidth=0.25)
    cmap = plt.get_cmap(cmap_)
    cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    for i in range(x.shape[1] - 2):
        plt.plot(
            x[t, 0], x[t, 2 + i], color=scalarMap.to_rgba(i))
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(my_formatter))
#    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(my_formatter))
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y4lim is not None:
        ax.set_ylim(y4lim)
    ax.set_xticks(ticks)

    plt.tight_layout() # NOTE: not sure if it works fine
    plt.savefig(prefix + '.pdf')
    plt.close()


if __name__ == "__main__":
    mpl.rc('lines', linewidth=2, markersize=8)
    mpl.rc('font', size=12)
    mpl.rc('grid', color='0.75', linestyle=':')
    mpl.rc('ps', useafm=True)  # Force to use
    mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
    mpl.rc('text', usetex=True)  # for a paper submision

    # plot_ddrcma("../dat4fig/m3f4d160debug")
    # plot_ddcma("../dat4fig/m0f4d160debug")
    # plot_ddrcma("../dat4fig/m10f4d160debug")

    # plot_ddrcma("../dat4fig/m3f8d160debug")
    # plot_ddcma("../dat4fig/m0f8d160debug")
    # plot_ddrcma("../dat4fig/m10f8d160debug")

    # plot_ddrcma("../dat4fig/m3f13d160debug")
    # plot_ddcma("../dat4fig/m0f13d160debug")
    # plot_ddrcma("../dat4fig/m10f13d160debug")

    # plot_ddrcma("../dat4fig/m4f7d160debug")
    plot_ddrcma("../dat4fig/m3f7d160debug", xlim=(-2e4*0.05, 2e4), y4lim=(1e-1, 3e3), nsample=2000)
    # plot_ddcma("../dat4fig/m0f7d160debug")
    # plot_ddrcma("../dat4fig/m10f7d160debug")

    # plot_ddrcma("../dat4fig/m3f11d160debug")
    # plot_ddcma("../dat4fig/m0f11d160debug")
    # plot_ddrcma("../dat4fig/m10f11d160debug")

    # plot_ddrcma("../dat4fig/m4f15d160debug")
    # plot_ddcma("../dat4fig/m0f15d160debug")
    # plot_ddrcma("../dat4fig/m10f15d160debug")

    # plot_ddrcma("../dat4fig/m3f12d160debug")
    # plot_ddcma("../dat4fig/m0f12d160debug")
    # plot_ddrcma("../dat4fig/m10f12d160debug")
    # plot_ddrcma("../dat/m3f2d320debug")