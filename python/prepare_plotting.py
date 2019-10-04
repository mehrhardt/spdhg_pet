iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a))


epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in algs}

out = misc.resort_out(out, obj_opt)

# set line width and style
lwidth = 1
lwidth_help = 1
lstyle = '-'
lstyle_help = '--'
# set colors using colorbrewer
bmap = brewer2mpl.get_map('Set1', 'Qualitative', 5)
colors = bmap.mpl_colors
colors = list(colors)

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
# available markers:
# ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
markers = ('o', 'v', 's', 'p', 'd', '^', '*')
mevery = [(np.float(i) / 30, 1e-1) for i in range(20)]  # how many markers to draw
msize = 5

def get_ind(x, y, xlim, ylim):
    return (np.greater_equal(x, xlim[0]) & np.less_equal(x, xlim[1]) &
            np.greater_equal(y, ylim[0]) & np.less_equal(y, ylim[1]))