"""Compute ML results for
[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schoenlieb, 
Faster PET reconstruction with non-smooth priors by randomization and 
preconditioning, Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07"""
from __future__ import print_function, division
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter

folder_data_amyloid = '/home/me404/store/data/201611_PET_Pawel_amyloid'
folder_data_fdg = '/home/me404/store/data/201712_PET_Pawel_fdg'
folder_out = '/home/me404/store/projects/201804_SPDHG_PET/results'
folder_file = '/home/me404/store/repositories/gitbb_spdhg_pawel/python'
folder_odl = '/home/me404/store/repositories/github_myODL'

import sys
sys.path.append(folder_odl)

import misc
import mMR
from stochastic_primal_dual_hybrid_gradient import pdhg, spdhg

import odl
from odl.contrib import fom
from odl.solvers import CallbackPrintIteration, CallbackPrintTiming

#%% set parameters and create folder structure
filename = 'ml'

nepoch = 30
nepoch_target = 5000
datasets = ['fdg', 'amyloid10min']

tol_step = 1e-6
rho = 0.999

folder_norms = '{}/norms'.format(folder_out)
misc.mkdir(folder_norms)

for dataset in datasets:

    if dataset is 'amyloid10min':
        folder_data = folder_data_amyloid
        planes = None
        data_suffix = 'rings0-64_span1_time3000-3600'
        clim = [0, 1]  # set colour limit for plots

    elif dataset is 'fdg':
        folder_data = folder_data_fdg
        planes = [85, 90, 46]
        data_suffix = 'rings0-64_span1'
        clim = [0, 10]  # set colour limit for plots

    folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
    misc.mkdir(folder_main)
    misc.mkdir('{}/pics'.format(folder_main))
    misc.mkdir('{}/py'.format(folder_main))
    misc.mkdir('{}/logs'.format(folder_main))

    folder_today = '{}/nepochs{}'.format(folder_main, nepoch)
    misc.mkdir(folder_today)
    misc.mkdir('{}/npy'.format(folder_today))
    misc.mkdir('{}/pics'.format(folder_today))
    misc.mkdir('{}/figs'.format(folder_today))

    # load real data
    file_data = '{}/data_{}.npy'.format(folder_data, data_suffix)
    data, background, factors, image, image_mr, image_ct = np.load(file_data)

    # convert to odl
    Y = mMR.operator_mmr().range
    factors = Y.element(factors)
    data = Y.element(data)
    background = Y.element(background)

    # define operator
    K = mMR.operator_mmr(factors=factors)
    X = K.domain
    KL = misc.kullback_leibler(Y, data, background)
    obj_fun = KL * K
 
    # set smoothing
    fwhm = np.array([4, 4, 4])  # in mm
    sd_smoothing = fwhm / (2 * np.sqrt(2 * np.log(2)) * X.cell_sides)

    def smoothing(x):
        return X.element(gaussian_filter(x.asarray(), sigma=sd_smoothing))

    def save_image(x, n, f):
        misc.save_image(x.asarray(), n, f, planes=planes, clim=clim)

        xs = smoothing(x)
        n = 'smoothed_{}'.format(n)
        misc.save_image(xs.asarray(), n, f, planes=planes, clim=clim)

    if not os.path.exists('{}/pics/gray_image_pet.png'.format(folder_main)):
        tmp = X.element()
        fldr = '{}/pics'.format(folder_main)
        K.toodl(image, tmp)
        misc.save_image(tmp.asarray(), 'image_pet', fldr, planes=planes)
        K.toodl(image_mr, tmp)
        misc.save_image(tmp.asarray(), 'image_mr', fldr, planes=planes)
        K.toodl(image_ct, tmp)
        misc.save_image(tmp.asarray(), 'image_ct', fldr, planes=planes)

    # %% --- get target --- BE CAREFUL, THIS TAKES TIME
    file_target = '{}/target.npy'.format(folder_main)
    if not os.path.exists(file_target):
        print('file {} does not exist. Compute it.'.format(file_target))
        x_opt = X.one()
        misc.MLEM(x_opt, KL.data, KL.background, K, nepoch_target, 
                  verbose=True)

        obj_opt = obj_fun(x_opt)
        x_opt_smoothed = smoothing(x_opt)

        save_image(x_opt, 'target', '{}/pics'.format(folder_main))

        np.save(file_target, (x_opt, obj_opt, x_opt_smoothed))

    else:
        print('file {} exists. Load it.'.format(file_target))
        x_opt, obj_opt, x_opt_smoothed = np.load(file_target)

    # define a function to compute statistic during the iterations
    class CallbackStore(odl.solvers.Callback):

        def __init__(self, alg, iter_save, iter_plot, niter_per_epoch):
            self.iter_save = iter_save
            self.iter_plot = iter_plot
            self.iter_count = 0
            self.alg = alg
            self.out = []
            self.niter_per_epoch = niter_per_epoch

        def __call__(self, x, Kx=None, tmp=None, **kwargs):

            if type(x) is list:
                x = x[0]

            k = self.iter_count

            if k in self.iter_save:
                if Kx is None:
                    Kx = K(x)

                x_smoothed = smoothing(x)
                obj = KL(Kx, tmp=tmp)
                psnr_opt = fom.psnr(x, x_opt)
                psnr_opt_smoothed = fom.psnr(x_smoothed, x_opt_smoothed)

                self.out.append({'obj': obj, 'psnr_opt': psnr_opt,
                                 'psnr_opt_smoothed': psnr_opt_smoothed})

            if k in self.iter_plot:
                save_image(x, '{}_{}'.format(self.alg,
                                             int(k / self.niter_per_epoch)),
                           '{}/pics'.format(folder_today))

            self.iter_count += 1

    # set number of subsets for algorithms
    nsub = {'MLEM': 1, 'OSEM-21': 21, 'OSEM-100': 100, 'COSEM-252': 252,
            'SPDHG2-21': 21, 'SPDHG2-100': 100, 'SPDHG2-252': 252}

    # %% run algorithms
    algs = nsub.keys()

    for alg in algs:
        file_result = '{}/npy/{}.npy'.format(folder_today, alg)

        if os.path.exists(file_result):
            print('file {} does exist. Do NOT compute it.'.format(file_result))
        else:
            print('file {} does not exist. Compute it.'.format(file_result))

            if nsub[alg] > 1:
                partition = mMR.partition_by_angle(nsub[alg])
                Ys = mMR.operator_mmr(sino_partition=partition).range
                fctrs = Ys.element([factors[s, :] for s in partition])
                d = Ys.element([data[s, :] for s in partition])
                bg = Ys.element([background[s, :] for s in partition])

                # define operator
                Ks = mMR.operator_mmr(factors=fctrs, sino_partition=partition)
                KLs = misc.kullback_leibler(Ys, d, bg)  # data fit

            prob = [1 / nsub[alg]] * nsub[alg]
            niter_per_epoch = int(np.round(nsub[alg] / sum(prob)))
            niter = nepoch * niter_per_epoch
            iter_save, iter_plot = misc.what_to_save(niter_per_epoch, nepoch)

            # output function to be used with the iterations
            step = 1
            cb = (CallbackPrintIteration(step=step, end=', ') &
                  CallbackPrintTiming(step=step, cumulative=False, end=', ') &
                  CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                      cumulative=True) &
                  CallbackStore(alg, iter_save, iter_plot,
                                niter_per_epoch))

            x = X.one()  # initialise variable
            cb(x)

            if alg.startswith('SPDHG') or alg.startswith('PDHG'):
                g = odl.solvers.functional.IndicatorBox(X, lower=X.zero())

            if alg.startswith('MLEM'):
                misc.MLEM(x, KL.data, KL.background, K, niter, callback=cb)

            elif alg.startswith('OSEM'):
                misc.OSEM(x, KLs.data, KLs.background, Ks, niter, callback=cb)

            elif alg.startswith('COSEM'):
                misc.COSEM(x, KLs.data, KLs.background, Ks, niter, callback=cb)

            elif alg.startswith('PDHG1'):
                norm_K = misc.norm(K, '{}/norm_1subset.npy'
                                   .format(folder_norms))
                sigma = rho / norm_K
                tau = rho / norm_K
                f = KL
                A = K
                
                pdhg(x, f, g, A, tau, sigma, niter, callback=cb)

            elif alg.startswith('SPDHG1'):
                norm_K = misc.norms(Ks, '{}/norm_{}subsets.npy'.format(
                                        folder_norms, nsub[alg]))
                sigma = [rho / nk for nk in norm_K]
                tau = rho / (len(Ks) * max(norm_K))
                f = KLs
                A = Ks

                spdhg(x, f, g, A, tau, sigma, niter, callback=cb)

            elif alg.startswith('PDHG2'):
                f = KL
                A = K

                one = A.domain.one()
                tmp = A.range.element()
                A(one, out=tmp)
                tmp.ufuncs.maximum(tol_step, out=tmp)
                sigma = rho / tmp

                one = A.range.one()
                tmp = A.domain.element()
                A.adjoint(one, out=tmp)
                tmp.ufuncs.maximum(tol_step, out=tmp)
                tau = rho / tmp

                pdhg(x, f, g, A, tau, sigma, niter, callback=cb)

            elif alg.startswith('SPDHG2'):
                f = KLs
                A = Ks

                one = A.domain.one()
                tmp = A.range.element()
                A(one, out=tmp)
                tmp.ufuncs.maximum(tol_step, out=tmp)
                sigma = rho / tmp

                tmp = A.domain.element()
                max_domain = A.domain.zero()
                for i in range(len(A)):
                    one = A[i].range.one()
                    A[i].adjoint(one, out=tmp)
                    tmp.ufuncs.maximum(max_domain, out=max_domain)
                max_domain.ufuncs.maximum(tol_step, out=max_domain)
                tau = rho / (len(A) * max_domain)

                spdhg(x, f, g, A, tau, sigma, niter, callback=cb)

            else:
                raise NameError('Algorithm not defined')

            np.save(file_result, (iter_save, niter, niter_per_epoch, x,
                                  cb.callbacks[1].out, nsub[alg], prob))

    # %% show all methods
    iter_save_v, out_v, niter_per_epoch_v = {}, {}, {}
    for a in algs:
        (iter_save_v[a], _, niter_per_epoch_v[a], _, out_v[a], _, _
         ) = np.load('{}/npy/{}.npy'.format(folder_today, a))

    out = misc.resort_out(out_v, obj_opt)
    misc.quick_visual_output(iter_save_v, algs, out, niter_per_epoch_v,
                             folder_today)