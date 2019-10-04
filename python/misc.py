"""Auxiliary functions"""

from __future__ import print_function, division

import odl
from odl.solvers.util.callback import (CallbackPrintIteration,
                                       CallbackPrintTiming)

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable


def norms(ops, file_norm):
    if not os.path.exists(file_norm):
        print('file {} does not exist. Compute it.'.format(file_norm))
        rnd = odl.phantom.uniform_noise(ops[0].domain)
        norm_ops = [1.05 * odl.power_method_opnorm(op, maxiter=100, xstart=rnd)
                    for op in ops]
        np.save(file_norm, norm_ops)
    else:
        print('file {} exists. Load it.'.format(file_norm))
        norm_ops = np.load(file_norm)

    return norm_ops


def norm(op, file_norm):
    if not os.path.exists(file_norm):
        print('file {} does not exist. Compute it.'.format(file_norm))
        rnd = odl.phantom.uniform_noise(op.domain)
        norm_op = 1.05 * odl.power_method_opnorm(op, maxiter=100, xstart=rnd)
        np.save(file_norm, norm_op)
    else:
        print('file {} exists. Load it.'.format(file_norm))
        norm_op = np.load(file_norm)

    return norm_op


def save_image(image, name, folder, fignum=1, cmaps={'gray', 'inferno'},
               clim=None, planes=None):

    if planes is None:
        planes = [i // 2 for i in image.shape]

    for cmap in cmaps:
        fig = plt.figure(fignum)
        plt.clf()
        imagesc3(image, clim=clim, cmap=cmap, title=name, planes=planes)
        fig.savefig('{}/{}_{}.png'.format(folder, cmap, name),
                    bbox_inches='tight')

        if clim is None:
            x = image - np.min(image)
            if np.max(x) > 1e-4:
                x /= np.max(x)
        else:
            x = (image - clim[0]) / (clim[1] - clim[0])

        x = np.minimum(np.maximum(x, 0), 1)

        flnm = '{}/{}_x{}_{}.png'.format(folder, cmap, planes[0], name)
        imsave(flnm, x[planes[0], :, :].T, cmap=cmap, vmin=0, vmax=1)

        flnm = '{}/{}_y{}_{}.png'.format(folder, cmap, planes[1], name)
        imsave(flnm, x[:, planes[1], :].T, cmap=cmap, vmin=0, vmax=1)

        flnm = '{}/{}_z{}_{}.png'.format(folder, cmap, planes[2], name)
        imsave(flnm, x[:, :, planes[2]], cmap=cmap, vmin=0, vmax=1)


def what_to_save(niter_per_epoch, nepoch):
    first_part = min(nepoch, 1)
    second_part = min(nepoch, 5)
    iter_save = list(np.unique(np.ceil(np.array(
        list(np.arange(0, first_part, .2)) +
        list(np.arange(first_part, second_part, .5)) +
        list(np.arange(second_part, nepoch + 1))) *
                 niter_per_epoch).astype('int')))
    iter_plot = list(np.unique(np.ceil(np.array(
        [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]) *
                 niter_per_epoch).astype('int')))
    return iter_save, iter_plot


def resort_out(out, obj_opt, verbose=False):
    print('resort_out')

    algs = out.keys()

    out_resorted = {}
    out_resorted['obj_opt'] = obj_opt
    for a in algs:
        if verbose:
            print('=== {}'.format(a))
        out_resorted[a] = {}
        K = len(out[a])

        for meas in out[a][0].keys():  # quality measures
            if verbose:
                print('    === ' + meas)
            if np.isscalar(out[a][0][meas]):
                out_resorted[a][meas] = np.nan * np.ones(K)

                for k in range(K):  # iterations
                    out_resorted[a][meas][k] = out[a][k][meas]

        meas = 'obj_rel'
        if verbose:
            print('    === {}'.format(meas))
        out_resorted[a][meas] = np.nan * np.ones(K)

        for k in range(K):  # iterations
            out_resorted[a][meas][k] = ((out[a][k]['obj'] - obj_opt) /
                                        (out[a][0]['obj'] - obj_opt))

    for a in algs:  # algorithms
        if verbose:
            print('=== {}'.format(a))
        K = len(out[a])
        for meas in out_resorted[a].keys():  # quality measures
            if verbose:
                print('   === {}'.format(meas))
            for k in range(K):  # iterations
                if out_resorted[a][meas][k] <= 0:
                    out_resorted[a][meas][k] = np.nan

    return out_resorted


def quick_visual_output(iter_save, algs, out, niter_per_epoch, folder_today):
    print('quick_visual_output')

    epoch_save = {a: np.array(iter_save[a]) / np.float(niter_per_epoch[a])
                  for a in algs}

    prefix = ''
    prefix = now() + '_'

    fig = plt.figure(1)
    plt.clf()

    plots = out[algs[0]].keys()
    logy = ['obj', 'obj_rel']

    for plotx in ['linx', 'logx']:
        for meas in plots:
            print('=== {} === {} ==='.format(plotx, meas))
            plt.clf()

            if plotx == 'linx':
                if meas in logy:
                    for alg in algs:
                        x = epoch_save[alg]
                        y = out[alg][meas]
                        plt.semilogy(x, y, linewidth=3, label=alg)
                else:
                    for alg in algs:
                        x = epoch_save[alg]
                        y = out[alg][meas]
                        plt.plot(x, y, linewidth=3, label=alg)

            elif plotx == 'logx':
                if meas in logy:
                    for alg in algs:
                        x = epoch_save[alg][1:]
                        y = out[alg][meas][1:]
                        plt.loglog(x, y, linewidth=3, label=alg)
                else:
                    for alg in algs:
                        x = epoch_save[alg][1:]
                        y = out[alg][meas][1:]
                        plt.semilogx(x, y, linewidth=3, label=alg)

            plt.title('{} v iteration'.format(meas))
            h = plt.gca()
            h.set_xlabel('epochs')
            plt.legend(loc='best')

            fig.savefig('{}/pics/{}{}_{}.png'.format(folder_today, prefix,
                        meas, plotx), bbox_inches='tight')


def get_ind(x, y, xlim, ylim):
    return (np.greater_equal(x, xlim[0]) & np.less_equal(x, xlim[1]) &
            np.greater_equal(y, ylim[0]) & np.less_equal(y, ylim[1]))
  
def plots_paper(figs, ylims, yticks, epoch_save, out, lstyles, 
                algs, colors, labels, legendy, ncols):
    
    import matplotlib
    
    # set latex options
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsfonts}']
    
    # set font
    wsize = (7, 3)
    fsize = 11
    font = {'family': 'sans-serif', 'size': fsize}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', labelsize=fsize)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fsize)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fsize)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fsize)    # legend fontsize
    
    # set markers
    markers = ('o', 'v', 's', 'p', 'd', '^', '*')
    mevery = [(np.float(i) / 30, 1e-1) for i in range(20)]  # how many markers to draw
    msize = 5

    figs.append(plt.figure(figsize=wsize, dpi=200))
    plt.clf()
    
    plt.subplot(121)
    meas = 'psnr_opt'
    xlim = [0, 30]

    for j, alg in enumerate(algs):
        x = epoch_save[alg].copy()
        y = out[alg][meas]
        i = get_ind(x, y, xlim, ylims)
        plt.plot(x[i], y[i], color=colors[j], linestyle=lstyles[j],
                 linewidth=1, marker=markers[j], markersize=msize,
                 markevery=mevery[j], label=labels[j])
    
    plt.gca().set_ylabel('PSNR (to minimizer)')
    plt.ylim(ylims)
    plt.xlim(xlim)
    if yticks is not None:
        plt.gca().yaxis.set_ticks(yticks)
    
    plt.subplot(122)
    meas = 'obj_rel'
    
    for j, alg in enumerate(algs):
        x = epoch_save[alg].copy()
        y = out[alg][meas]
        i = get_ind(x, y, xlim, [1e-10, 2])
        plt.semilogy(x[i], y[i], color=colors[j], linestyle=lstyles[j],
                     linewidth=1, marker=markers[j], markersize=msize,
                     markevery=mevery[j], label="__noentry__")
    
    plt.text(-25, 2e-5, 'epochs = expected number of forward projections')
    
    plt.gca().set_ylabel('rel. objective')
    plt.ylim([1e-4, 1])
    plt.xlim(xlim)
    plt.figlegend(frameon=False, loc='upper center', fancybox=True, framealpha=0.5, 
                  ncol=ncols, bbox_to_anchor=(0.5, legendy))
    
    plt.tight_layout()
    plt.show()

def plots_paper_psnr(figs, ylims, yticks, epoch_save, out, lstyles, 
                     algs, colors, labels, legendy, ncols):
    
    import matplotlib
    
    # set latex options
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsfonts}']
    
    # set font
    wsize = (5, 3)
    fsize = 11
    font = {'family': 'sans-serif', 'size': fsize}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', labelsize=fsize)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fsize)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fsize)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fsize)    # legend fontsize
    
    # set markers
    markers = ('o', 'v', 's', 'p', 'd', '^', '*')
    mevery = [(np.float(i) / 30, 1e-1) for i in range(20)]  # how many markers to draw
    msize = 5

    figs.append(plt.figure(figsize=wsize, dpi=200))
    plt.clf()
    
    meas = 'psnr_opt'
    xlim = [0, 30]

    for j, alg in enumerate(algs):
        x = epoch_save[alg].copy()
        y = out[alg][meas]
        i = get_ind(x, y, xlim, ylims)
        plt.plot(x[i], y[i], color=colors[j], linestyle=lstyles[j],
                 linewidth=1, marker=markers[j], markersize=msize,
                 markevery=mevery[j], label=labels[j])
    
    plt.gca().set_xlabel('epochs = expected number of forward projections')
    plt.gca().set_ylabel('PSNR (to minimizer)')
    plt.ylim(ylims)
    plt.xlim(xlim)
    if yticks is not None:
        plt.gca().yaxis.set_ticks(yticks)
    
    plt.legend(frameon=True, loc='best', fancybox=True, framealpha=.5, 
               ncol=ncols)
    
    plt.tight_layout()
    plt.show()
    
    
def colors_set():
    import brewer2mpl
    colors = list(brewer2mpl.get_map('Set1', 'Qualitative', 5).mpl_colors)
    colors.pop(3)
    return colors


def colors_pair():
    import brewer2mpl
    return list(brewer2mpl.get_map('Paired', 'Qualitative', 10).mpl_colors)


def colors_sequ():
    import brewer2mpl
    colors = list(brewer2mpl.get_map('YlGn', 'Sequential', 5).mpl_colors)
    colors.pop(0)
    return colors


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def now():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M')



def imagesc(imInput, clim=None, fig=None, cmap='gray', subplot='111',
            interpolation='none', title=None, colorbar=True, axis=None,
            aspect='auto', clf=False, **kwargs):
    """
    Shows images similar to MATLABs imagesc function.

    Arguments:
    ----------
    data: ndarray
        A data cube. Third dimension should be the image index.
    clim: list (optional)
        minimal and maximal colour values to clip. They may be None.
    fig: figure handle (optional)
        A figure handle where the image will be displayed. If not set, a new
        figure will be created.
    cmap: string (optional)
        colormap to be used {gray, hot, ...}
    subplot : string (optional)
        specifies the subplot this image should be plotted in
    interpolation : string (optional)
        interpolation as a string {none, ...}
    title : string (optional)
        string to be plotted as the title
    colorbar : boolean (optional)
        shall a colorbar be plotted next to the images?
    clf: boolean (optional)
        if true, erase the content of the current figure
    kwargs : Keyword arguments (optional)
        Other keyword arguments are passed to imshow.

    Returns:
    --------
    fig : figure handle
        handle to the figure that has been plotted to
    h_im : subplot handle
        handle to the image axes that has been plotted to
    """

    if fig is None:
        fig = plt.gcf()
    elif fig == "new":
        fig = plt.figure()

    if clf or subplot == '111' or subplot == (1, 1, 1):
        fig.clear()

    if clim is None:
        clim = [None, None]

    if clim[0] is None:
        clim[0] = np.min(imInput)

    if clim[1] is None:
        clim[1] = np.max(imInput)

    if title is None:
        title = ""

    if isinstance(subplot, str):
        ax = fig.add_subplot(subplot)
    else:
        ax = fig.add_subplot(*subplot)

    sbplt = ax.imshow(imInput, interpolation=interpolation,
                      aspect=aspect, clim=clim, **kwargs)

    ax.set_title(title)
    sbplt.set_cmap(cmap)

    if colorbar:
        divider = make_axes_locatable(ax)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(sbplt, cax=cax)

    if axis == 'off':
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if axis == 'image':
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')

    if axis == 'equal':
        ax.set_aspect('equal')

    if axis == 'auto':
        ax.set_aspect('auto')

    fig.canvas.draw()

    return fig, sbplt, ax


def imagesc3(imInput, clim=None, fig=None, cmap='gray', planes=None,
             interpolation='none', title=None, subtitles=None, aspect='auto',
             **kwargs):
    """
    Shows images similar to MATLABs imagesc function.

    Arguments:
    ----------
    data : ndarray
        A data cube. Third dimension should be the image index.
    fig : figure handle (optional)
        A figure handle where the image will be displayed. If not set, a new
        figure will be created.
    cmap : string (optional)
        colormap to be used {gray, hot, ...}
    planes : ndarray (optional)
        A tuple of three numbers indicating which slices should be shown.
        If not specified, the centre slices are shown.
    interpolation : string (optional)
        interpolation as a string {none, ...}
    title : string (optional)
        string to be plotted as the title
    subtitles : array of strings (optional)
        strings to be plotted as titles of the subfigures
    kwargs : Keyword arguments (optional)
        Other keyword arguments are passed to imshow.

    Returns:
    --------
    fig : figure handle
        handle to the figure that has been plotted to
    h_im : subplot handle
        handle to the image axes that has been plotted to
    """

    if fig is None:
        fig = plt.gcf()
    elif fig == "new":
        fig = plt.figure()

    if clim is None:
        clim = [np.min(imInput), np.max(imInput)]

    if planes is None:
        planes = np.int32(np.ceil(np.array(imInput.shape)/2))

    if title is None:
        title = ""

    if aspect == "auto" or np.isscalar(aspect):
        aspect = [aspect, aspect, aspect]

    if subtitles is None:
        subtitles = ("", "", "")
    else:
        for i in range(len(subtitles)):
            subtitles[i] = ": " + subtitles[i]

    _, sbplt1, ax1 = imagesc(
        imInput[:, :, int(planes[2])], clim, fig, cmap, '231', interpolation,
        'z='+str(int(planes[2]))+subtitles[0], False, None, aspect[0], False,
        **kwargs)
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')

    _, sbplt2, ax2 = imagesc(
        np.transpose(imInput[:, planes[1], :]), clim, fig, cmap, '232',
        interpolation,
        'y='+str(int(planes[1]))+subtitles[1], False, None, aspect[1], False,
        **kwargs)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')

    _, sbplt3, ax3 = imagesc(
        np.transpose(imInput[planes[0], :, :]), clim, fig, cmap, '234',
        interpolation,
        'x='+str(int(planes[0]))+subtitles[2], False, None, aspect[2],
        False, **kwargs)
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')

    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    plt.colorbar(sbplt3, cax=cbaxes)

    plt.suptitle('3D plot: {}'.format(title))
    fig.canvas.draw()

    return fig


def kullback_leibler(Y, data, background):

    try:
        if not isinstance(data, Y):
            data = Y.element(data)
    except TypeError:
        data = Y.element(data)

    try:
        if not isinstance(background, Y):
            background = Y.element(background)
    except TypeError:
        background = Y.element(background)

    if isinstance(Y, odl.ProductSpace):
        f = odl.solvers.SeparableSum(
                *[KullbackLeibler(Yi, yi, ri)
                  for (Yi, yi, ri) in zip(Y, data, background)])
        f.data = data
        f.background = background

        return f
    else:
        return KullbackLeibler(Y, data, background)


class KullbackLeibler(odl.solvers.Functional):

    """The Kullback-Leibler divergence functional.

    Notes
    -----
    The functional :math:`F` with prior :math:`g>0` is given by:

    .. math::
        F(x)
        =
        \\begin{cases}
            \\sum_{i} \left( x_i - g_i + g_i \log \left( \\frac{g_i}{ x_i }
            \\right) \\right) & \\text{if } x_i > 0 \\forall i
            \\\\
            +\\infty & \\text{else.}
        \\end{cases}

    KL based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.

    The functional is related to the Kullback-Leibler cross entropy functional
    `KullbackLeiblerCrossEntropy`. The KL cross entropy is the one
    diescribed in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, and
    the functional :math:`F` is obtained by switching place of the prior and
    the varialbe in the KL cross entropy functional.

    For a theoretical exposition, see `Csiszar1991`_.

    See Also
    --------
    KullbackLeiblerConvexConj : the convex conjugate functional
    KullbackLeiblerCrossEntropy : related functional

    References
    ----------
    .. _Csiszar1991:  http://www.jstor.org/stable/2241918
    """

    def __init__(self, space, data, background):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        data : ``space`` `element-like`
            Data term, non-negative.
        background : ``space`` `element-like`
            Background term, positive.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        if background not in self.domain:
            raise ValueError('`background` not in `domain`'
                             ''.format(background, self.domain))

        # TODO: Check that data is nonnegative and background is positive
        self.__data = data
        self.__background = background
        self.__offset = None

    @property
    def data(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The prior in convex conjugate Kullback-Leibler functional."""
        return self.__background

    def offset(self, tmp=None):
        """The offset which is independent of the unknown.

        Needs one extra array of memory of the size of `prior`.
        """

        Y = self.domain

        if self.__offset is None:
            if tmp is None:
                tmp = Y.element()

            # define short variable names
            y = self.data
            r = self.background

            # Compute
            #   sum(r - y + y * log(y)) with 0 log 0 = 0.
            # Note that for any integer y
            #   y * log(y) = y * log(max(y, 1)).
            tmp = Y.element(np.maximum(y.asarray(), 1))
            tmp.ufuncs.log(out=tmp)
            tmp *= y
            tmp.lincomb(1, tmp, 1, r)
            tmp.lincomb(1, tmp, -1, y)

            # sum the result up
            self.__offset = tmp.ufuncs.sum()

        return self.__offset

    def __call__(self, x, tmp=None):
        """Return the KL-diveregnce in the point ``x``.

        If any components of ``x`` is non-positive, the value is positive
        infinity.

        Needs one extra array of memory of the size of `prior`.


        Examples
        --------
        >>> X = odl.rn(1)
        >>> data = X.element(2)
        >>> background = X.element(1)
        >>> KL = KullbackLeibler(X, data, background)
        >>> x = X.element(1)
        >>> KL(x)
        0

        >>> data = X.element(0)
        >>> KL = KullbackLeibler(X, data, background)
        >>> KL(x)
        2
        """

        Y = self.domain

        if tmp is None:
            tmp = Y.element()

        # define short variable names
        y = self.data
        r = self.background

        # Compute
        #   sum(x + r - y + y * log(y / (x + r)))
        # = sum(x - y * log(x + r)) + self.offset
        # Assume that
        #   x + r > 0

        # Naive:
        # tmp = ((x - y + y * (y / x)).ufuncs.log()
        #        .inner(self.domain.one()))

        tmp.lincomb(1, x, 1, r)
        tmp.ufuncs.log(out=tmp)
        tmp *= y
        tmp.lincomb(1, x, -1, tmp)

        # sum the result up
        obj = tmp.ufuncs.sum() + self.offset(tmp=tmp)

        if np.isnan(obj):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return obj

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The gradient is not defined in points where one or more components
        are non-positive.
        """
        functional = self

        class KLGradient(odl.Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.
                The gradient is not defined in points where one or more
                components are non-positive.
                """

                # TODO: Probably not correct any more.
                return (-functional.data) / x + 1

        return KLGradient()

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj :
            Proximal of the convex conjugate of a functional.
        """
        return odl.solvers.nonsmooth.proximal_operators.proximal_cconj(
                odl.solvers.nonsmooth.proximal_operators.proximal_cconj_kl(
                        space=self.domain, g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerConvexConj(self.domain, self.data,
                                         self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.domain, self.data,
                                             self.background)


class KullbackLeiblerConvexConj(odl.solvers.Functional):

    """The convex conjugate of Kullback-Leibler divergence functional.

    Notes
    -----
    The functional :math:`F^*` with prior :math:`g>0` is given by:

    .. math::
        F^*(x)
        =
        \\begin{cases}
            \\sum_{i} \left( -g_i \ln(1 - x_i) \\right)
            & \\text{if } x_i < 1 \\forall i
            \\\\
            +\\infty & \\text{else}
        \\end{cases}

    See Also
    --------
    KullbackLeibler : convex conjugate functional
    """

    def __init__(self, space, data, background):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        data : ``space`` `element-like`
            Data term, non-negative.
        background : ``space`` `element-like`
            Background term, positive.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        if background not in self.domain:
            raise ValueError('`background` not in `domain`'
                             ''.format(background, self.domain))

        self.__data = data
        self.__background = background

    @property
    def data(self):
        """The prior in convex conjugate Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The prior in convex conjugate Kullback-Leibler functional."""
        return self.__background

    def _call(self, x):
        """Return the value in the point ``x``.

        If any components of ``x`` is larger than or equal to 1, the value is
        positive infinity.
        """

        y = self.data
        
        tmp = (-y * (1 - x).funcs.log()).inner(self.domain.one())

        if np.isnan(tmp):
            # In this case, some element was larger than or equal to one
            return np.inf
        else:
            return tmp

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The gradient is not defined in points where one or more components
        are larger than or equal to one.
        """
        functional = self

        class KLCCGradient(odl.Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.

                The gradient is not defined in points where one or more
                components are larger than or equal to one.
                """

                raise NotImplementedError

                if functional.prior is None:
                    return 1.0 / (1 - x)
                else:
                    return functional.data / (1 - x)

        return KLCCGradient()

    def proximal(self, sigma):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj :
            Proximal of the convex conjugate of a functional.
        """

        class ProximalCConjKL(odl.Operator):

            """Proximal operator of the convex conjugate of the KL divergence."""

            def __init__(self, sigma, data, background):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                """
                self.sigma = sigma
                self.data = data
                self.background = background
                super().__init__(domain=data.space, range=data.space,
                     linear=False)

            def _call(self, x, out):
                """
                Examples
                --------
                >>> X = odl.rn(1)
                >>> data = X.element(4)
                >>> background = X.element(1)
                >>> KL = KullbackLeibler(X, data, background)
                >>> x = X.element(1)
                >>> KL.convex_conj.proximal(2)(x)
                X.element(-1)

                >>> data = X.element(0)
                >>> KL = KullbackLeibler(X, data, background)
                >>> KL.convex_conj.proximal(2)(x)
                X.element(1)
                """

                # Let y = data, r = background, z = x + s * r
                # Compute 0.5 * (z + 1 - sqrt((z - 1)**2 + 4 * s * y))
                # Currently it needs 3 extra copies of memory.

                # define short variable names
                y = self.data
                r = self.background
                s = self.sigma

                # Compute
                #   sum(x + r - y + y * log(y / (x + r)))
                # = sum(x - y * log(x + r)) + self.offset
                # Assume that
                #   x + r > 0

                z = self.domain.element()
                z.assign(r)

                if np.size(s) == 1:
                    z.lincomb(s, z, 1, x)
                else:
                    z *= s
                    z += x

                # compute sqrt
                out.assign(z)
                out -= 1
                out.ufuncs.square(out=out)

                if np.size(s) == 1:
                    out.lincomb(1, out, 4 * s, y)

                else:
                    # compute sigma * y
                    tmp = self.domain.element()
                    tmp.assign(y)
                    tmp *= s
                    out.lincomb(1, out, 4, tmp)

                    del(tmp)

                out.ufuncs.sqrt(out=out)

                # out = 0.5 * (z + 1 - sqrt(...))
                out.lincomb(1, z, -1, out)

                del(z)

                out += 1
                out /= 2.

        return ProximalCConjKL(sigma, self.data, self.background)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the conjugate KL-functional."""
        return KullbackLeibler(self.domain, self.data, self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.domain, self.data,
                                             self.background)
        

def MLEM(x, data, background, A, niter, **kwargs):
    """
    Computes the MLEM estimate with a fixed number of iterations.

    Arguments:
    ----------
    x : ndarray
        initital image
    data : np.array
        data, vector with non-negative components
    background : np.array
        background data, vector with positive components
    A : Operator
        forward operator
    niter : int
        number of iterations
    sens : np.array (optional)
        sensitivity image

    Returns:
    --------
    x
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Reduce to default callback for minimal verbosity
    verbose = kwargs.pop('verbose', False)
    if verbose and callback is None:
        callback = default_callback()

    memory = kwargs.pop('memory', 'high')
    sens = kwargs.pop('sens', None)

    eps = float(np.finfo(x.dtype).eps)

    if sens is None:
        one_range = A.range.one()
        sens = A.adjoint(one_range)
        del(one_range)

        eps_domain = 1e+4 * eps * A.domain.one()
        sens.ufuncs.maximum(eps_domain, out=sens)
        del(eps_domain)

    # allocate memory (this could be improved)
    tmp_domain = A.domain.element()
    if memory == 'high':
        Ax = A.range.element()
    tmp_range = A.range.element()

    # OSEM iteration
    for k in range(niter):
        if memory == 'high':
            A(x, out=Ax)
            tmp_range.lincomb(1, Ax, 1, background)
        else:
            A(x, out=tmp_range)
            tmp_range += background

        data.divide(tmp_range, out=tmp_range)
        A.adjoint(tmp_range, out=tmp_domain)

        x *= tmp_domain
        x /= sens

        if callback is not None:
            callback(x)


def OSEM(x, data, background, A, niter, sens=None, fun_select=None,
         callback=None, verbose=False):
    """
    Computes the MLEM estimate with a fixed number of iterations and ordered
    subset acceleration.

    Arguments:
    ----------
    x : np.array
        initital image
    data : list[np.array]
        data, vector with non-negative components
    background : list[np.array]
        background data, vector with positive components
    op : list[Operator]
        array of forward operators
    n_iter : int
        number of iterations
    sens : list[np.array] (optional)
        sensitivity image
    fun_select : function (optional)
        sequence how the subsets should be used.

    Returns:
    --------
    x
    """

    if verbose and callback is None:
        callback = default_callback()

    # sensitivity
    sens = (A.domain ** len(A)).element()
    for i in range(len(A)):
        one_range = A[i].range.one()
        A[i].adjoint(one_range, out=sens[i])
        del(one_range)

    eps = float(np.finfo(x.dtype).eps)
    eps_domain = 1e+4 * eps * A.domain.one()
    sens.ufuncs.maximum(eps_domain, out=sens)
    del(eps_domain)

    if fun_select is None:
        fun_select = default_fun_select(len(A))

    # allocate memory (this could be improved)
    tmp_domain = A.domain.element()

    # OSEM iteration
    for k in range(niter):
        # select an element
        i = fun_select(k)

        # MLEM update for subset
        tmp_range = A[i].range.element()
        A[i](x, out=tmp_range)
        tmp_range += background[i]
        data[i].divide(tmp_range, out=tmp_range)
        A[i].adjoint(tmp_range, out=tmp_domain)
        del(tmp_range)

        x *= tmp_domain
        x /= sens[i]

        if callback is not None:
            callback(x)


def COSEM(x, data, background, A, niter, sens=None, fun_select=None,
          callback=None, verbose=False):
    """
    Computes the MLEM estimate with a fixed number of iterations and a
    convergent ordered subset acceleration.

    Arguments:
    ----------
    x: np.array
        initital image
    data: np.array
        data, vector with non-negative components
    background: np.array
        background data, vector with positive components
    op: list[Operator]
        array of forward operators
    n_iter: int
        number of iterations
    sens: list (optional)
        sensitivity image
    fun_select: function (optional)
        sequence how the subsets should be used.

    Returns:
    --------
    x
    """

    if verbose and callback is None:
        callback = default_callback()

    # sensitivity
    sens = A.domain.element()
    one_range = A.range.one()
    A.adjoint(one_range, out=sens)
    del(one_range)

    eps = float(np.finfo(x.dtype).eps)
    eps_domain = 1e+4 * eps * A.domain.one()
    sens.ufuncs.maximum(eps_domain, out=sens)
    del(eps_domain)

    if fun_select is None:
        fun_select = default_fun_select(len(A))

    # allocate memory
    tmp_domain = A.domain.element()

    y = (A.domain ** len(A)).element()
    w = A.domain.zero()
    for i in range(len(A)):
        # compute Ai_adj(yi / (Ai(x) + ri))
        tmp_range = A[i].range.element()
        A[i](x, out=tmp_range)
        tmp_range += background[i]
        data[i].divide(tmp_range, out=tmp_range)
        A[i].adjoint(tmp_range, out=tmp_domain)
        del(tmp_range)

        # compute ynew = x * tmp and w += ynew
        y[i].assign(x)
        y[i] *= tmp_domain
        w += y[i]

    # COSEM iteration
    for k in range(niter):
        # select an element
        i = fun_select(k)

        # update image and other variables
        x.assign(w / sens)

        # compute Ai_adj(yi / (Ai(x) + ri))
        tmp_range = A[i].range.element()
        A[i](x, out=tmp_range)
        tmp_range += background[i]
        data[i].divide(tmp_range, out=tmp_range)
        A[i].adjoint(tmp_range, out=tmp_domain)
        del(tmp_range)

        # compute ynew = x * tmp and w += ynew - yold
        w -= y[i]
        tmp_domain *= x
        y[i].assign(tmp_domain)
        w += y[i]

        if callback is not None:
            callback(x)


def default_fun_select(n):
    r = np.random.permutation(n)

    def fun_select(k):
        return r[np.mod(k, n)]

    return fun_select

def default_callback(step=1):
    return (CallbackPrintIteration(step=step, end=', ') &
            CallbackPrintTiming(step=step, cumulative=False,
                                fmt='elapsed = {:5.03f} s', end=', ') &
            CallbackPrintTiming(step=step, fmt='total = {:5.03f} s',
                                cumulative=True))

