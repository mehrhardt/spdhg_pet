"""Compute TV results for
[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schoenlieb, 
Faster PET reconstruction with non-smooth priors by randomization and 
preconditioning, Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07"""
from __future__ import print_function, division
import os
import numpy as np
from shutil import copyfile

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
filename = 'map_tv'

nepoch = 30
nepoch_target = 5000
datasets = ['fdg', 'amyloid10min']
datasets = ['fdg']

rho = 0.999
tol_step = 1e-6

folder_norms = '{}/norms'.format(folder_out)
misc.mkdir(folder_norms)

for dataset in datasets:
    print('<<< ' + dataset)

    if dataset == 'amyloid10min':
        folder_data = folder_data_amyloid
        planes = None
        alphas = [5]
        clim = [0, 1]  # colour limits for plots
        data_suffix = 'rings0-64_span1_time3000-3600'

    elif dataset == 'fdg':
        folder_data = folder_data_fdg
        planes = [85, 90, 46]
        alphas = [1.2]
        clim = [0, 10]  # colour limits for plots
        data_suffix = 'rings0-64_span1'

    def save_image(x, n, f):
        misc.save_image(x.asarray(), n, f, planes=planes, clim=clim)

    folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
    misc.mkdir(folder_main)
    misc.mkdir('{}/py'.format(folder_main))
    misc.mkdir('{}/logs'.format(folder_main))

    # copy file
    copyfile('{}/{}.py'.format(folder_file, filename),
             '{}/py/{}_{}.py'.format(folder_main, misc.now(), filename))

    # load real data and convert to odl
    file_data = '{}/data_{}.npy'.format(folder_data, data_suffix)
    (data, background, factors, image, image_mr,
     image_ct) = np.load(file_data)
    Y = mMR.operator_mmr().range
    data = Y.element(data)
    background = Y.element(background)
    factors = Y.element(factors)

    # define operator
    K = mMR.operator_mmr(factors=factors)
    X = K.domain
    norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))

    KL = misc.kullback_leibler(Y, data, background)

    for alpha in alphas:
        print('<<< <<< alpha = {}'.format(alpha))

        folder_param = '{}/alpha{:.2g}'.format(folder_main, alpha)
        misc.mkdir(folder_param)
        misc.mkdir('{}/pics'.format(folder_param))

        folder_today = '{}/nepochs{}'.format(folder_param, nepoch)
        misc.mkdir(folder_today)
        misc.mkdir('{}/npy'.format(folder_today))
        misc.mkdir('{}/pics'.format(folder_today))
        misc.mkdir('{}/figs'.format(folder_today))

        D = odl.Gradient(X)
        norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))

        c = norm_K / norm_D
        D = odl.Gradient(X) * c
        norm_D *= c
        L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)
        L164 = (alpha / c) * odl.solvers.GroupL1Norm(D.range.astype('float64'))
        g = odl.solvers.IndicatorBox(X, lower=0)

        obj_fun = KL * K + L1 * D + g  # objective functional
 
        if not os.path.exists('{}/pics/gray_image_pet.png'
                              .format(folder_param)):
            tmp = X.element()
            tmp_op = mMR.operator_mmr()
            tmp_op.toodl(image, tmp)
            fldr = '{}/pics'.format(folder_param)
            misc.save_image(tmp.asarray(), 'image_pet', fldr, planes=planes)
            tmp_op.toodl(image_mr, tmp)
            misc.save_image(tmp.asarray(), 'image_mr', fldr, planes=planes)
            tmp_op.toodl(image_ct, tmp)
            misc.save_image(tmp.asarray(), 'image_ct', fldr, planes=planes)

        # --- get target --- BE CAREFUL, THIS TAKES TIME
        file_target = '{}/target.npy'.format(folder_param)
        if not os.path.exists(file_target):
            print('file {} does not exist. Compute it.'.format(file_target))

            A = odl.BroadcastOperator(K, D)
            f = odl.solvers.SeparableSum(KL, L1)

            norm_A = misc.norm(A, '{}/norm_tv.npy'.format(folder_main))
            sigma = rho / norm_A
            tau = rho / norm_A

            niter_target = nepoch_target

            step = 10
            cb = (CallbackPrintIteration(step=step, end=', ') &
                  CallbackPrintTiming(step=step, cumulative=False, end=', ') &
                  CallbackPrintTiming(step=step, cumulative=True,
                                      fmt='total={:.3f} s'))

            x_opt = X.zero()
            odl.solvers.pdhg(x_opt, g, f, A, niter_target, tau, sigma,
                             callback=cb)

            obj_opt = obj_fun(x_opt)

            save_image(x_opt, 'target', '{}/pics'.format(folder_param))
            np.save(file_target, (x_opt, obj_opt))
        else:
            print('file {} exists. Load it.'.format(file_target))
            x_opt, obj_opt = np.load(file_target)

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
                    obj = obj_fun(x)
                    psnr_opt = fom.psnr(x, x_opt)

                    self.out.append({'obj': obj, 'psnr_opt': psnr_opt})

                if k in self.iter_plot:
                    save_image(x, '{}_{}'.format(self.alg,
                                                 int(k / niter_per_epoch)),
                               '{}/pics'.format(folder_today))

                self.iter_count += 1

        # set number of subsets for algorithms
        nsub = {'PDHG1': 1, 'PDHG2': 1,
                'SPDHG1-21-uni': 21, 'SPDHG1-100-uni': 100, 
                'SPDHG1-252-uni': 252, 
                'SPDHG1-21-bal': 21, 'SPDHG1-100-bal': 100, 
                'SPDHG1-252-bal': 252, 
                'SPDHG2-21-uni': 21, 'SPDHG2-100-uni': 100, 
                'SPDHG2-252-uni': 252, 
                'SPDHG2-21-bal': 21, 'SPDHG2-100-bal': 100, 
                'SPDHG2-252-bal': 252}

        # %% run algorithms
        algs = nsub.keys()

        for alg in algs:
            file_result = '{}/npy/{}.npy'.format(folder_today, alg)

            if os.path.exists(file_result):
                print('file {} does exist. Do NOT compute it.'
                      .format(file_result))
            else:
                print('file {} does not exist. Compute it.'
                      .format(file_result))

                # define operator for subsets
                if nsub[alg] > 1:
                    partition = mMR.partition_by_angle(nsub[alg])
                    tmp = mMR.operator_mmr(sino_partition=partition)
                    Ys = tmp.range
                    fctrs = Ys.element([factors[s, :] for s in partition])
                    Ks = mMR.operator_mmr(factors=fctrs, 
                                          sino_partition=partition)

                    d = Ys.element([data[s, :] for s in partition])
                    bg = Ys.element([background[s, :] for s in partition])
                    KLs = misc.kullback_leibler(Ys, d, bg)

                    norm_Ks = misc.norms(Ks, '{}/norm_{}subsets.npy'
                                             .format(folder_norms, nsub[alg]))

                    A = odl.BroadcastOperator(*(list(Ks.operators) + [D]))
                    functionals = (list(KLs.functionals) + [L1])
                    f = odl.solvers.SeparableSum(*functionals)
                    norm_Ai = list(norm_Ks) + [float(norm_D)]

                else:
                    A = odl.BroadcastOperator(K, D)
                    f = odl.solvers.SeparableSum(KL, L1)
                    norm_Ai = [norm_K, norm_D]

                if alg.endswith('uni'):
                    prob = [1 / len(A)] * len(A)
                elif alg.endswith('bal'):
                    prob = [0.5 / nsub[alg]] * nsub[alg] + [0.5]
                elif alg.endswith('imp'):
                    prob = [nAi / sum(norm_Ai) for nAi in norm_Ai]
                else:
                    prob = [1, 1]

                niter_per_epoch = int(np.round(nsub[alg] / sum(prob[:-1])))
                niter = nepoch * niter_per_epoch
                iter_save, iter_plot = misc.what_to_save(niter_per_epoch,
                                                         nepoch)

                # output function to be used with the iterations
                step = int(np.ceil(niter_per_epoch / 10))
                cb = (CallbackPrintIteration(step=step, end=', ') &
                      CallbackPrintTiming(step=step, cumulative=False,
                                          end=', ') &
                      CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                          cumulative=True) &
                      CallbackStore(alg, iter_save, iter_plot,
                                    niter_per_epoch))

                x = X.zero()  # initialise variable
                cb(x)

                if alg.startswith('PDHG1'):
                    norm_A = misc.norm(A, '{}/norm_tv.npy'
                                       .format(folder_main))
                    sigma = rho / norm_A
                    tau = rho / norm_A

                    pdhg(x, f, g, A, tau, sigma, niter, callback=cb)

                elif alg.startswith('SPDHG1'):
                    sigma = [rho / nAi for nAi in norm_Ai]
                    tau = rho * min([pi / nAi
                                     for pi, nAi in zip(prob, norm_Ai)])

                    spdhg(x, f, g, A, tau, sigma, niter, prob=prob, callback=cb)

                elif alg.startswith('PDHG2'):
                    one = A[0].domain.one()
                    tmp = A[0].range.element()
                    A[0](one, out=tmp)
                    tmp.ufuncs.maximum(tol_step, out=tmp)
                    sigma = [rho / tmp, rho / norm_D]

                    one = A[0].range.one()
                    tmp = A[0].domain.element()
                    A[0].adjoint(one, out=tmp)
                    tmp.ufuncs.maximum(tol_step, out=tmp)
                    tmp.ufuncs.maximum(norm_D, out=tmp)
                    tau = (0.5 * rho) / tmp

                    def fun_select(x):
                        return [0, 1]

                    spdhg(x, f, g, A, tau, sigma, niter, prob=[1, 1],
                          fun_select=fun_select, callback=cb)

                elif alg.startswith('SPDHG2'):
                    one = A.domain.one()
                    tmp = A.range.element()
                    A(one, out=tmp)
                    tmp.ufuncs.maximum(tol_step, out=tmp)
                    sigma = [rho / t for t in tmp[:-1]] + [rho / norm_D]

                    tmp = A.domain.element()
                    max_domain = tol_step * A.domain.one()
                    for pi, Ai in zip(prob[:-1], A[:-1]):
                        one = Ai.range.one()
                        Ai.adjoint(one, out=tmp)
                        tmp /= pi
                        tmp.ufuncs.maximum(max_domain, out=max_domain)
                    max_domain.ufuncs.maximum(norm_D / prob[-1],
                                              out=max_domain)

                    tau = rho / max_domain

                    spdhg(x, f, g, A, tau, sigma, niter, prob=prob,
                          callback=cb)

                else:
                    raise NameError('Algorithm not defined')

                np.save(file_result, (iter_save, niter, niter_per_epoch, x,
                                      cb.callbacks[1].out, nsub[alg], prob))

        # %%  show all methods
        iter_save_v, out_v, niter_per_epoch_v = {}, {}, {}
        for a in algs:
            (iter_save_v[a], _, niter_per_epoch_v[a], _, out_v[a], _, _
             ) = np.load('{}/npy/{}.npy'.format(folder_today, a))

        out = misc.resort_out(out_v, obj_opt)
        misc.quick_visual_output(iter_save_v, algs, out, niter_per_epoch_v,
                                 folder_today)