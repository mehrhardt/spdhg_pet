"""Print figures of parameters and side information for 
[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schoenlieb, 
Faster PET reconstruction with non-smooth priors by randomization and 
preconditioning, Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave

folder_data_amyloid = '/home/me404/store/data/201611_PET_Pawel_amyloid'
folder_data_fdg = '/home/me404/store/data/201712_PET_Pawel_fdg'
folder_out = '/home/me404/store/projects/201804_SPDHG_PET/results'
folder_file = '/home/me404/store/repositories/gitbb_spdhg_pawel/python'
folder_odl = '/home/me404/store/repositories/github_myODL'

import sys
sys.path.append(folder_odl)

import mMR
import misc

import odl

#%% set parameters and create folder structure
filename = 'map_dtv'

eta = 1e-6
datasets = ['amyloid', 'fdg']

for dataset in datasets:
    print('<<< ' + dataset)

    if dataset is 'amyloid':
        folder_data = folder_data_amyloid
        planes = None

    elif dataset is 'fdg':
        folder_data = folder_data_fdg
        planes = [85, 90, 46]

    folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
    misc.mkdir(folder_main)
    misc.mkdir('{}/py'.format(folder_main))
    misc.mkdir('{}/logs'.format(folder_main))

    folder_param = '{}/eta{:.2g}'.format(folder_main, eta)
    misc.mkdir(folder_param)
    misc.mkdir('{}/pics'.format(folder_param))

    # load real data and convert to odl
    data_suffix = 'rings0-64_span1'
    file_data = '{}/data_{}.npy'.format(folder_data, data_suffix)
    (data, background, factors, image, image_mr,
     image_ct) = np.load(file_data)
    Y0 = mMR.operator_mmr().range
    factors = Y0.element(factors)

    # define operator
    K = mMR.operator_mmr(factors=factors)
    X = K.domain

    gradient = odl.Gradient(X)

    sideinfo = X.element()
    K.toodl(image_mr, sideinfo)
    sideinfo_grad = gradient(sideinfo)

    gradient_space = gradient.range
    norm = odl.PointwiseNorm(gradient_space, 2)
    norm_sideinfo_grad = norm(sideinfo_grad)

    max_norm = np.max(norm_sideinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sideinfo_grad = np.sqrt(norm_sideinfo_grad ** 2 +
                                     eta_scaled ** 2)
    xi = gradient_space.element([g / norm_eta_sideinfo_grad
                                 for g in sideinfo_grad])

    fldr = '{}/pics'.format(folder_param)

    tmp = norm(sideinfo_grad)
    misc.save_image(tmp.asarray(), 'image_norm_sideinfo_grad',
                    fldr, planes=planes, cmaps={'gray'}, clim=[0, 100])

    tol_step = 1e-6

    one = K.domain.one()
    tmp = K.range.element()
    K(one, out=tmp)
    tmp.ufuncs.maximum(tol_step, out=tmp)
    sigma = 1 / tmp

    one = K.range.one()
    tmp = K.domain.element()
    K.adjoint(one, out=tmp)
    tmp.ufuncs.maximum(tol_step, out=tmp)
    tau = 1 / tmp

    misc.save_image(tau.asarray(), 'tau',
                    fldr, planes=planes, cmaps={'gray'}, clim=[0, .005])

    image = sigma
    name = 'sigma'
    clim = [0, .1]

    def print2d(image, name, clim):
        cmap = 'gray'
        fig = plt.figure(0)
        plt.clf()
        misc.imagesc(image, clim=clim, cmap=cmap, title=name)

        fig.savefig('{}/{}_{}.png'.format(fldr, cmap, name),
                    bbox_inches='tight')

        if clim is None:
            x = image - np.min(image)
            if np.max(x) > 1e-4:
                x /= np.max(x)
        else:
            x = (image - clim[0]) / (clim[1] - clim[0])

        x = np.minimum(np.maximum(x, 0), 1)

        flnm = '{}/{}_{}.png'.format(fldr, cmap, name)

        imsave(flnm, x, cmap=cmap, vmin=0, vmax=1)

    print2d(sigma, 'sigma', [0, 0.1])

    tmp = K.putgaps(sigma)

    misc.save_image(tmp, 'sigma2', fldr, cmaps={'gray'}, clim=[0, .2])

    for i in [100, 1000, 2000, 3000]:
        print2d(tmp[i, ...], 'sigma_sino{}'.format(i), [0, 0.2])