"""Print figures for
[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schoenlieb, 
Faster PET reconstruction with non-smooth priors by randomization and 
preconditioning, Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07"""
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')

folder_odl = '/home/me404/store/repositories/github_myODL'
folder_out = '/home/me404/store/repositories/gitbb_SPDHG_Pawel/latex/pics'
folder_in = '/home/me404/store/projects/201804_SPDHG_PET/results'

import sys
sys.path.append(folder_odl)

import misc

#%%
formats = ['png']
filename = 'ml'
all_algs = ['MLEM', 'OSEM-21', 'OSEM-100', 'COSEM-252', 'SPDHG2-21', 'SPDHG2-100', 'SPDHG2-252']

nepoch = 30
dataset = 'fdg'
dataset = 'amyloid10min'

folder_main = '{}/{}_{}'.format(folder_in, filename, dataset)
folder_today = '{}/nepochs{}'.format(folder_main, nepoch)

file_target = '{}/target.npy'.format(folder_main)
x_opt, obj_opt, x_opt_smoothed = np.load(file_target, allow_pickle=True)

iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in all_algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a),
                                 allow_pickle=True)

epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in all_algs}

out = misc.resort_out(out, obj_opt)

# draw figures
fig = []
algs = ['MLEM', 'OSEM-21', 'COSEM-252', 'SPDHG2-252']
labels = ['MLEM', 'OSEM', 'COSEM', r'SPDHG+$^\ast$']
lstyles = 3 * ['--'] + 1 * ['-']

misc.plots_paper_psnr(fig, [15, 55], [20, 30, 40, 50], epoch_save, out, 
                      lstyles, algs, misc.colors_set(), labels, 1.1, 2)

algs = ['OSEM-21', 'OSEM-100', 'SPDHG2-21', 'SPDHG2-100']
labels = ['OSEM (21 subsets)', 'OSEM (100)', 
          r'SPDHG+$^\ast$ (21)', r'SPDHG+$^\ast$ (100)']
lstyles = 2 * ['--'] + 2 * ['-']

misc.plots_paper_psnr(fig, [15, 55], [20, 30, 40, 50], epoch_save, out, 
                      lstyles, algs, misc.colors_pair(), labels, 1.1, 1)
#
for fmt in formats:
    for i, fi in enumerate(fig):
        fi.savefig('{}/{}_{}_{}_out{}.{}'.format(folder_out, fmt,
                   filename, dataset, i, fmt), bbox_inches='tight')
        plt.close(fi)

#%%
filename = 'map_tv'
all_algs = ['PDHG1', 'PDHG2', 
            'SPDHG1-21-uni', 'SPDHG1-21-bal', 'SPDHG2-21-uni',
            'SPDHG1-100-uni', 'SPDHG2-100-uni', 'SPDHG1-100-bal', 
            'SPDHG1-252-uni', 'SPDHG1-252-bal']

nepoch = 30
dataset = 'fdg'
alpha = 1.2

folder_main = '{}/{}_{}'.format(folder_in, filename, dataset)
folder_alpha = '{}/alpha{:.2g}'.format(folder_main, alpha)
folder_today = '{}/nepochs{}'.format(folder_alpha, nepoch)

file_target = '{}/target.npy'.format(folder_alpha)
x_opt, obj_opt = np.load(file_target, allow_pickle=True)

iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in all_algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a), 
                                 allow_pickle=True)

epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in all_algs}

out = misc.resort_out(out, obj_opt)

fig = []
algs = ['PDHG1', 'SPDHG1-21-uni', 'SPDHG1-100-uni', 'SPDHG1-252-uni']
labels = ['PDHG', 'SPDHG (21 subsets)', 'SPDHG (100)', 
          'SPDHG (252)']
lstyles = ['--'] + 3 * ['-']

misc.plots_paper(fig, [19, 35], [20, 25, 30, 35], epoch_save, out, 
                 lstyles, algs, misc.colors_sequ(), labels, 1.1, 4)
  
algs = ['SPDHG1-21-uni', 'SPDHG1-21-bal', 
        'SPDHG1-100-uni', 'SPDHG1-100-bal', 
        'SPDHG1-252-uni', 'SPDHG1-252-bal']
labels = ['21 subsets, uniform sampling', '21, balanced', 
          '100, uniform', '100, balanced',
          '252, uniform', '252, balanced']
lstyles = ['-'] * 6

misc.plots_paper(fig, [19, 48], [20, 30, 40], epoch_save, out, 
                 lstyles, algs, misc.colors_pair(), labels, 1.2, 3)

algs = ['PDHG1', 'PDHG2', 'SPDHG1-21-uni', 'SPDHG2-21-uni', 
        'SPDHG1-100-uni', 'SPDHG2-100-uni']
labels = ['PDHG', 'PDHG (precond)', 'SPDHG (21 subsets)', 'SPDHG (21, precond)', 
          'SPDHG (100)', 'SPDHG (100, precond)']
lstyles = ['--'] * 2 + ['-'] * 4

misc.plots_paper(fig, [19, 35], [20, 25, 30, 35], epoch_save, out, 
                 lstyles, algs, misc.colors_pair(), labels, 1.2, 3)

#
for fmt in formats:
    for i, fi in enumerate(fig):
        fi.savefig('{}/{}_{}_{}_out{}.{}'.format(folder_out, fmt,
                   filename, dataset, i, fmt), bbox_inches='tight')
        plt.close(fi)

#%%
filename = 'map_atv'
all_algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']

nepoch = 30
dataset = 'amyloid10min'
alpha = 3

folder_main = '{}/{}_{}'.format(folder_in, filename, dataset)
folder_alpha = '{}/alpha{:.2g}'.format(folder_main, alpha)
folder_today = '{}/nepochs{}'.format(folder_alpha, nepoch)

file_target = '{}/target.npy'.format(folder_alpha)
x_opt, obj_opt = np.load(file_target, allow_pickle=True)

iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in all_algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a), 
                                 allow_pickle=True)

epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in all_algs}

out = misc.resort_out(out, obj_opt)

fig = []
algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']
labels = ['PDHG', 'SPDHG+ (21 subsets)', 'SPDHG+ (100)', 'SPDHG+ (252)']
lstyles = ['--'] + 3 * ['-']

misc.plots_paper(fig, [15, 40], [20, 30, 40], epoch_save, out, 
                 lstyles, algs, misc.colors_sequ(), labels, 1.1, 4)

for fmt in formats:
    for i, fi in enumerate(fig):
        fi.savefig('{}/{}_{}_{}_out{}.{}'.format(folder_out, fmt,
                   filename, dataset, i, fmt), bbox_inches='tight')
        plt.close(fi)
        
#%%
filename = 'map_dtv'
all_algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']

nepoch = 30
dataset = 'amyloid10min'
alpha = 10
eta = 1e-4

folder_main = '{}/{}_{}'.format(folder_in, filename, dataset)
folder_param = '{}/alpha{:.2g}_eta{:.2g}'.format(folder_main, alpha, eta)
folder_today = '{}/nepochs{}'.format(folder_param, nepoch)

file_target = '{}/target.npy'.format(folder_param)
x_opt, obj_opt = np.load(file_target, allow_pickle=True)

iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in all_algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a), 
                                 allow_pickle=True)

epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in all_algs}

out = misc.resort_out(out, obj_opt)

fig = []
algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']
labels = ['PDHG', 'SPDHG+ (21 subsets)', 'SPDHG+ (100)', 'SPDHG+ (252)']
lstyles = ['--'] + 3 * ['-']

misc.plots_paper(fig, [19, 47], [20, 30, 40], epoch_save, out, 
                 lstyles, algs, misc.colors_sequ(), labels, 1.1, 4)

for fmt in formats:
    for i, fi in enumerate(fig):
        fi.savefig('{}/{}_{}_{}_out{}.{}'.format(folder_out, fmt,
                   filename, dataset, i, fmt), bbox_inches='tight')
        plt.close(fi)

#%%
filename = 'map_tgv'
all_algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']

nepoch = 30
dataset = 'fdg'
alpha = 2
beta = .5

folder_main = '{}/{}_{}'.format(folder_in, filename, dataset)
folder_param = '{}/alpha{:.2g}_beta{:.2g}'.format(folder_main, alpha, beta)
folder_today = '{}/nepochs{}'.format(folder_param, nepoch)

file_target = '{}/target.npy'.format(folder_param)
x_opt, obj_opt = np.load(file_target, allow_pickle=True)

iter_save, niter, niter_per_epoch, image, out, nsub, prob = (
    {}, {}, {}, {}, {}, {}, {})
for a in all_algs:
    (iter_save[a], niter[a], niter_per_epoch[a], image[a], out[a],
     nsub[a], prob[a]) = np.load('{}/npy/{}.npy'.format(folder_today, a), 
                                 allow_pickle=True)

epoch_save = {a: np.array(iter_save[a]) /
              np.float(niter_per_epoch[a]) for a in all_algs}

out = misc.resort_out(out, obj_opt)

fig = []
algs = ['PDHG1', 'SPDHG2-21-bal', 'SPDHG2-100-bal', 'SPDHG2-252-bal']
labels = ['PDHG', 'SPDHG+ (21 subsets)', 'SPDHG+ (100)', 'SPDHG+ (252)']
lstyles = ['--'] + 3 * ['-']

misc.plots_paper(fig, [19, 61], [20, 30, 40, 50, 60], epoch_save, out, 
                 lstyles, algs, misc.colors_sequ(), labels, 1.1, 4)

for fmt in formats:
    for i, fi in enumerate(fig):
        fi.savefig('{}/{}_{}_{}_out{}.{}'.format(folder_out, fmt,
                   filename, dataset, i, fmt), bbox_inches='tight')
        plt.close(fi)