"""Compute TGV results for
[1] M. J. Ehrhardt, P. J. Markiewicz, and C.-B. Schoenlieb, 
Faster PET reconstruction with non-smooth priors by randomization and 
preconditioning, Phys. Med. Biol., 2019. 10.1088/1361-6560/ab3d07"""
from __future__ import print_function, division
import os
import numpy as np

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
filename = 'map_tgv'

nepoch = 30
nepoch_target = 5000
#nepoch_target = 500
datasets = ['fdg', 'amyloid10min']
datasets = ['fdg']

rho = 0.999  # algorithm parameter (rho < 1)
tol_step = 1e-6

folder_norms = '{}/norms'.format(folder_out)
misc.mkdir(folder_norms)

for dataset in datasets:
    print('<<< ' + dataset)

    if dataset == 'amyloid10min':
        folder_data = folder_data_amyloid
        planes = None
        alphas = [4, 5]
        betas = [0.5, 1]
        data_suffix = 'rings0-64_span1_time3000-3600'
        clim = [0, 1]  # colour limits for plots

    elif dataset == 'fdg':
        folder_data = folder_data_fdg
        planes = [85, 90, 46]
        alphas = [2]
        betas = [0.5]
        data_suffix = 'rings0-64_span1'
        clim = [0, 10]  # colour limits for plots

    folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
    misc.mkdir(folder_main)
    misc.mkdir('{}/py'.format(folder_main))
    misc.mkdir('{}/logs'.format(folder_main))

    for alpha, beta in [(a, b) for a in alphas for b in betas]:
        print('<<< <<< alpha = {}, beta = {}'.format(alpha, beta))

        folder_param = '{}/alpha{:.2g}_beta{:.2g}'.format(folder_main, alpha,
                                                          beta)
        misc.mkdir(folder_param)
        misc.mkdir('{}/pics'.format(folder_param))

        folder_today = '{}/nepochs{}'.format(folder_param, nepoch)
        misc.mkdir(folder_today)
        misc.mkdir('{}/npy'.format(folder_today))
        misc.mkdir('{}/pics'.format(folder_today))
        misc.mkdir('{}/figs'.format(folder_today))

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
        U = K.domain
        norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))

        KL = misc.kullback_leibler(Y, data, background)

        gradient = odl.Gradient(U)
        Id = odl.IdentityOperator(gradient.range)

        PD = [odl.PartialDerivative(U, i, method='backward',
                                    pad_mode='symmetric')
              for i in range(3)]

        E = odl.operator.ProductSpaceOperator(
            [[PD[0], 0, 0],
             [0, PD[1], 0],
             [0, 0, PD[2]],
             [PD[1], PD[0], 0],
             [PD[2], 0, PD[0]],
             [0, PD[2], PD[1]]])

        D = odl.ProductSpaceOperator([[gradient, -Id],
                                      [0, E]])
        norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))
        norm_vfield = odl.PointwiseNorm(gradient.range)

        def save_image(x, n, f):
            misc.save_image(x[0].asarray(), n, f, planes=planes, clim=clim)
            misc.save_image(norm_vfield(x[1]).asarray(), n + '_norm_vfield',
                            f, planes=planes)

        c = float(norm_K) / float(norm_D)
        D *= c
        norm_D *= c
        L1 = odl.solvers.SeparableSum(*[
                (alpha / c) * odl.solvers.GroupL1Norm(gradient.range),
                (alpha * beta / c) * odl.solvers.GroupL1Norm(E.range)])

        g = odl.solvers.SeparableSum(
                odl.solvers.IndicatorBox(U, lower=0),
                odl.solvers.ZeroFunctional(gradient.range))

        X = D.domain
        P = [odl.ComponentProjection(X, i) for i in range(2)]
        A_ = odl.BroadcastOperator(K * P[0], D)
        f_ = odl.solvers.SeparableSum(KL, L1)
        obj_fun = f_ * A_ + g  # objective functional

        fldr = '{}/pics'.format(folder_param)
        if not os.path.exists('{}/gray_image_pet.png'.format(fldr)):
            tmp = U.element()
            tmp_op = mMR.operator_mmr()
            tmp_op.toodl(image, tmp)
            misc.save_image(tmp.asarray(), 'image_pet', fldr, planes=planes)
            tmp_op.toodl(image_mr, tmp)
            misc.save_image(tmp.asarray(), 'image_mr', fldr, planes=planes)
            tmp_op.toodl(image_ct, tmp)
            misc.save_image(tmp.asarray(), 'image_ct', fldr, planes=planes)

        # --- get target --- BE CAREFUL, THIS TAKES TIME
        file_target = '{}/target.npy'.format(folder_param)
        if not os.path.exists(file_target):
            print('file {} does not exist. Compute it.'.format(file_target))

            A = A_
            f = f_

            norm_A = misc.norm(A, '{}/norm_tgv.npy'.format(folder_main))
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

                    psnr_opt = fom.psnr(x[0], x_opt[0])
                    diff_opt = (x[0] - x_opt[0]).norm() / x_opt[0].norm()
                    diff_opt_v = (x[1] - x_opt[1]).norm() / x_opt[1].norm()

                    self.out.append({'obj': obj, 'psnr_opt': psnr_opt,
                                     'diff_opt': diff_opt,
                                     'diff_opt_v': diff_opt_v})

                if k in self.iter_plot:
                    save_image(x, '{}_{}'.format(self.alg,
                                                 int(k / niter_per_epoch)),
                               '{}/pics'.format(folder_today))

                self.iter_count += 1

        # set number of subsets for algorithms
        nsub = {'PDHG1': 1, 'SPDHG2-21-bal': 21, 
                'SPDHG2-100-bal': 100, 'SPDHG2-252-bal': 252}

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
                    d = Ys.element([data[s, :] for s in partition])
                    bg = Ys.element([background[s, :] for s in partition])

                    Ks = mMR.operator_mmr(factors=fctrs, 
                                          sino_partition=partition)
                    KLs = misc.kullback_leibler(Ys, d, bg)

                    norm_Ks = misc.norms(Ks, '{}/norm_{}subsets.npy'
                                             .format(folder_norms, nsub[alg]))

                    list_K = [op * P[0] for op in Ks]
                    ops = list_K + [D]
                    A = odl.BroadcastOperator(*ops)
                    f = odl.solvers.SeparableSum(*(list(KLs.functionals) +
                                                   [L1]))
                    norm_Ai = list(norm_Ks) + [norm_D]

                else:
                    A = A_
                    f = f_
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
                step = 1
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
                    norm_A = misc.norm(A, '{}/norm_tgv.npy'
                                          .format(folder_main))
                    sigma = rho / norm_A
                    tau = rho / norm_A

                    pdhg(x, f, g, A, tau, sigma, niter, callback=cb)

                elif alg.startswith('SPDHG1'):
                    sigma = [rho / nAi for nAi in norm_Ai]
                    tau = rho * min([pi / nAi
                                     for pi, nAi in zip(prob, norm_Ai)])

                    spdhg(x, f, g, A, tau, sigma, niter, prob=prob,
                          callback=cb)

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
                    tau = (1. / 2 * rho) / tmp

                    def fun_select(x):
                        return [1, 1]

                    spdhg(x, f, g, A, tau, sigma, niter, fun_select=fun_select,
                          callback=cb)

                    tau, sigma = [None, ] * 2

                elif alg.startswith('SPDHG2'):
                    one = A.domain.one()
                    tmp = A.range.element()
                    A(one, out=tmp)
                    tmp.ufuncs.maximum(tol_step, out=tmp)
                    sigma = ([rho / t for t in tmp[:-1]] +
                             [rho / norm_D])

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