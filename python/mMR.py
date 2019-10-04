"""Auxiliary functions and classes for reconstruction of data from the 
   Siemens Biograph mMR with nipet."""

from __future__ import division
import numpy as np
import os
from builtins import super
import matplotlib.pyplot as plt

import nipet
import odl
import misc


def operator_mmr(span=1, rings=(0, 64), im_shape=None, sino_partition=None,
                 factors=None, odlcuda=False, gpu_index=None, memory=True):

    # get backend and precision
    if odlcuda:
        impl = 'cuda'
        dtype = 'float32'

    else:
        impl = 'numpy'
        dtype = 'float32'

    # get geometry
    geometry = get_geometry_mmr(span, rings)

    # process sinogram partition
    ind_dtype = 'int32'
    if sino_partition is None:
        # geometry of the scanner
        Cnt, txLUT, axLUT = geometry
        subsets = False
        sino_partition = np.arange(txLUT['Naw'], dtype=ind_dtype)
    else:
        if np.isscalar(sino_partition[0]):
            subsets = False
            sino_partition = np.int32(sino_partition)
        else:
            subsets = True
            sino_partition = [np.int32(s) for s in sino_partition]

    if im_shape is None:
        im_shape = (170, 170, 2 * (rings[1] - rings[0]) - 1)

    domain = get_domain_mmr(geometry, im_shape, impl, dtype)
    range = get_range_mmr(geometry, subsets, sino_partition, impl, dtype)

    # convert factors to range
    if factors not in range:
        if factors is not None:  # this may not be the right thing to do
            if np.isscalar(factors):
                factors = factors * range.one()
            else:
                factors_list = [factors[s, :] for s in sino_partition]
                if memory:
                    factors = range.element(factors_list)

    # create operators
    if subsets:
        if factors is not None:
            ops = [OperatorMmr(domain, Y, geometry, s, f, gpu_index, memory)
                   for Y, s, f in zip(range, sino_partition, factors)]
        else:
            ops = [OperatorMmr(domain, Y, geometry, s, None, gpu_index, memory)
                   for Y, s in zip(range, sino_partition)]

        op = odl.BroadcastOperator(*ops)

    else:
        op = OperatorMmr(domain, range, geometry, sino_partition, factors,
                         gpu_index, memory)

    return op


def get_geometry_mmr(span, rings):

    # the span of the data, related to compression
    if span not in [1, 11]:
        raise NameError('NameError, span {} is not supported.'.format(span))

    nrings = rings[1] - rings[0]

    if nrings < 64 and not span == 1:
        raise NameError('Error, reduced rings are only supported for span=1')

    # geometry of the scanner
    Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()

    nrings = rings[1] - rings[0]

    # size of image
    if nrings < 64:
        axLUT = reduce_rings(axLUT, Cnt, rings)
    else:
        Cnt['rSZ_IMZ'] = Cnt['SZ_IMZ']
        Cnt['rNSN1'] = Cnt['NSN1']
        Cnt['rNSN11'] = Cnt['NSN11']

    Cnt['SPN'] = span

    geometry = Cnt, txLUT, axLUT

    return geometry


def get_domain_mmr(geometry, im_shape, impl, dtype):

    # geometry of the scanner
    Cnt, txLUT, axLUT = geometry

    # size of image
    if im_shape is None:
        im_shape = (170, 170, Cnt['rSZ_IMZ'])

    # voxel size in mm
    size_voxel = (Cnt['SO_VXX'], Cnt['SO_VXY'], Cnt['SO_VXZ'])  # in cm
    size_voxel = np.array(size_voxel) * 10  # in mm

    # domain
    fov_shape = im_shape * size_voxel
    domain = odl.uniform_discr(-fov_shape / 2, fov_shape / 2, im_shape,
                               weighting=None, impl=impl, dtype=dtype)

    return domain


def get_range_mmr(geometry, subsets, sino_partition, impl, dtype):

    # geometry of the scanner
    Cnt, txLUT, axLUT = geometry

    nsino = Cnt['rNSN' + str(Cnt['SPN'])]

    if subsets:
        ranges = []

        for s in sino_partition:
            data_shape = (len(s), nsino)
            space = odl.uniform_discr([0, 0], data_shape, data_shape,
                                      impl=impl, weighting=None, dtype=dtype)
            ranges.append(space)

        range = odl.ProductSpace(*ranges)

    else:
        data_shape = (len(sino_partition), nsino)
        range = odl.uniform_discr([0, 0], data_shape, data_shape, impl=impl,
                                  weighting=None, dtype=dtype)

    return range


class OperatorMmr(odl.Operator):
    """Class for the xray transform using nipet."""

    def __init__(self, domain, range, geometry, sino_ind, factors=None,
                 gpu_index=None, memory=True):
        """
        Initialize a new instance.

        Arguments:
        ----------
        sImage : list, tuple, np.array
            Size of the input images
        iSinogram : list, tuple, np.array (optional)
            list of sinogram bins to compute during projection
        span : string (optional)
            span of the projector
        computeNorm : boolean (optional)
            Do you want to compute the norm of the operator?
        factor : float (optional)
            Factor, that can be used for instance to model the noise level.

        Returns:
        --------
            => constructor
        """

        self.geometry = geometry
        self.sino_ind = sino_ind
        self.factors = factors
        self.gpu_index = gpu_index
        self.memory = memory

        if gpu_index is not None:
            self.geometry[0]['DEVID'] = gpu_index

        super().__init__(domain, range, linear=True)
        self.__norm = None

    def _call(self, image, out):
        """
        Implementation of the forward operator.

        Arguments:
        ----------
        image : array
            Three dimensional image input

        Returns:
        --------
        sino : array
            output data, stored as a stack of sinograms
        """

        if image not in self.domain:
            raise TypeError('Type error: input is not in domain! {} != {}'
                            .format(image.space, self.domain))

        dtype = np.dtype('float32')
        if image.dtype.num is not dtype.num:
            print('Type error: image is of type {} and not of type {}'
                  .format(image.dtype.num, dtype.num))
            raise NameError('Type error: image is of type {} and not of '
                            'type {}'.format(image.dtype.num, dtype.num))

        tmp_im = self.fromodl(image)
        Cnt, txLUT, axLUT = self.geometry
        ind = self.sino_ind
        tmp_out = np.zeros(self.range.shape, dtype='float32')

        nipet.prj.petprj.fprj(tmp_out, tmp_im, txLUT, axLUT, ind, Cnt, 0)
        out[:] = tmp_out

        if self.factors is not None:
            if self.memory:
                out *= self.factors
            else:
                out *= self.range.element(self.factors)

        return out

    def project_attenuation(self, image_large, out=None):
        """
        Implementation of the forward operator.

        Arguments:
        ----------
        image : array
            Three dimensional image input

        Returns:
        --------
        sino : array
            output data, stored as a stack of sinograms
        """

        """
        Implementation of the forward operator.

        Arguments:
        ----------
        image : array
            Three dimensional image input

        Returns:
        --------
        sino : array
            output data, stored as a stack of sinograms
        """

        if out is None:
            out = self.range.zero()

        Cnt, txLUT, axLUT = self.geometry
        ind = self.sino_ind
        tmp_out = np.zeros(self.range.shape, dtype='float32')

        nipet.prj.petprj.fprj(tmp_out, image_large, txLUT, axLUT, ind, Cnt, 1)
        out[:] = tmp_out

        return out

    @property
    def adjoint(self):
        return OperatorMmrAdjoint(self)

    @property
    def nipet_shape(self):
        Cnt, txLUT, axLUT = self.geometry
        return np.array((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['rSZ_IMZ']))

    def fromodl(self, x):
        tmp = np.zeros(self.nipet_shape, dtype='float32')
        R = (self.nipet_shape - np.array(self.domain.shape)) // 2

        if R[0] > 0:
            tmp[R[0]:-R[0], R[1]:-R[1], :] = x
        else:
            tmp[:] = x

        return tmp

    def toodl(self, x, out):
        R = (self.nipet_shape - np.array(self.domain.shape)) // 2
        if R[0] > 0:
            out[:] = x[R[0]:-R[0], R[1]:-R[1], :]
        else:
            out[:] = x

    def putgaps(self, data, out=None):
        # TODO, check that data is float32
        Cnt, txLUT, axLUT = self.geometry

        if out is None:
            # preallocate sino with gaps
            shape = (Cnt['NSANGLES'], Cnt['NSBINS'], self.range.shape[1])
            out = np.zeros(shape, dtype='float32')

        data = data.asarray().astype('float32')
        d = np.zeros((txLUT["Naw"], self.range.shape[1]), dtype='float32')
        d[self.sino_ind, :] = data

        # fill the sino with gaps
        nipet.mmr_auxe.pgaps(out, d, txLUT, Cnt)
        out = np.transpose(out, (2, 0, 1))

        return out

    def remgaps(self, sino, out=None):
        # TODO, check that sino is float32
        if out is None:
            # preallocate output sino without gaps, always in float
            out = np.zeros(self.range.shape, dtype='float32')

        Cnt, txLUT, axLUT = self.geometry

        sino = np.array(sino, dtype='float32')

        # fill the sino with gaps
        nipet.mmr_auxe.rgaps(out, sino, txLUT, Cnt)

        return out



class OperatorMmrAdjoint(odl.Operator):
    """Class for the adjoint of the xray transform using nipet."""

    def __init__(self, adjoint):

        super().__init__(adjoint.range, adjoint.domain, linear=True)
        self.__adjoint = adjoint

    def _call(self, y, out):
        """Backprojection

        Parameters
        ----------
        y : odlelement / ndarray
            Two dimensional sinogram

        Returns:
        --------
        backprojected image
        """

        adj = self.adjoint
        dtype = np.dtype('float32')
        if y.dtype.num is not dtype.num:
            raise TypeError('Type error: y is of type {}({}) and not of '
                            'type {}({})'
                            .format(y.dtype, y.dtype.num, dtype, dtype.num))

        if y not in self.domain:
            raise TypeError('Type error: input is not in domain! {} != {}'
                            .format(y.space, self.domain))

        y = y.copy()

        if adj.factors is not None:
            if adj.memory:
                y *= adj.factors
            else:
                y *= self.domain.element(adj.factors)

        Cnt, txLUT, axLUT = adj.geometry
        im = np.zeros(adj.nipet_shape, dtype='float32')
        ind = adj.sino_ind

        nipet.prj.petprj.bprj(im, y, txLUT, axLUT, ind, Cnt)

        adj.toodl(im, out)

        return out

    def norm(self):
        return self.adjoint.norm

    @property
    def adjoint(self):
        return self.__adjoint


def sampling_template(op=None):
    if op is None:
        Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
    else:
        Cnt, txLUT, axLUT = op.geometry

    rings = (0, 1)
    axLUT = reduce_rings(axLUT, Cnt, rings)
    data = np.arange(68516).reshape((-1, 1)).astype('float32')

    return nipet.mmraux.putgaps(data, txLUT, Cnt)[0, :, :].astype('int32')


def sino2ind(t, op=None):
    if op is None:
        op = operator_mmr(rings=(0, 1))

    tt = np.expand_dims(t, 0)
    tt = op.remgaps(tt)

    return tt[tt > -1].astype('int32')


def ind2sino(t, op=None):
    if op is None:
        op = operator_mmr(rings=(0, 1))

    y = op.range.zero().asarray()
    y[t, :] = 1
    y = op.range.element(y)

    return op.putgaps(y)[0, :, :]


def indices_full():
    txLUT = get_geometry_mmr(1, (0, 1))[0]

    return np.arange(txLUT['Naw'], dtype='int32')


def partition_by_angle(nsub, op=None):
    if op is None:
        op = operator_mmr(rings=(0, 1))

    s = sampling_template(op)
    n_angles = s.shape[0]
    isub = []
    for i in range(nsub):
        sc = s.copy()
        j = range(i, n_angles, nsub)
        mask = np.ones(n_angles, np.bool)
        mask[j] = False
        sc[mask, :] = -1

        isub.append(sino2ind(sc, op))

    return isub


def partition_by_bin(nsub, op=None, distribution='equidistant'):
    if op is None:
        op = operator_mmr(rings=(0, 1))

    s = sampling_template(op)
    isub = []
    nbin = s.shape[1]

    if distribution == 'equidistant':
        for i in range(nsub):
            sc = s.copy()
            j = range(i, nbin, nsub)
            mask = np.ones(nbin, np.bool)
            mask[j] = False
            sc[:, mask] = -1

            isub.append(sino2ind(sc, op))

    elif distribution == 'block':
        nbin_per_sub = nbin // nsub
        n_larger_subsets = nbin - nbin_per_sub * nsub

        j0 = 0
        for i in range(nsub):
            sc = s.copy()
            j1 = j0 + nbin_per_sub
            if i < n_larger_subsets:
                j1 += 1
            mask = np.ones(nbin, np.bool)
            mask[range(j0, j1)] = False
            j0 = j1
            sc[:, mask] = -1

            isub.append(sino2ind(sc, op))

    else:
        raise NameError('Not yet implemented')

    return isub


def partition_random(nsub, op=None):
    if op is None:
        op = operator_mmr(rings=(0, 1))

    s = sampling_template(op)
    t = sino2ind(s, op)
    tp = np.random.permutation(t.size)

    isub = list()
    for i in range(nsub):
        isub.append(tp[i::nsub])

    return isub


def reduce_rings(axLUT, Cnt, rings=(0, 64)):
    '''
    REDUCED RINGS
    offers customised axial FOV with reduced rings and therefore faster execution times
    rings included in the new, reduced projector are range(rs,re)
    works only with span-1
    '''

    Cnt['SPN'] = 1
    # select the number of sinograms for the number of rings
    # RNG_STRT is included in detection
    # RNG_END is not included in detection process
    Cnt['RNG_STRT'] = rings[0]
    Cnt['RNG_END'] = rings[1]
    nrings = rings[1] - rings[0]

    # number of axial voxels
    Cnt['rSO_IMZ'] = 2 * nrings - 1
    Cnt['rSZ_IMZ'] = 2 * nrings - 1
    # number of rings customised for the given ring range (only optional in span-1)
    Cnt['rNRNG'] = nrings
    # number of reduced sinos in span-1
    rNSN1 = nrings ** 2
    # correct for the limited max. ring difference in the full axial extent.
    # don't use ring range (1,63) as for this case no correction
    if nrings == 64:
        rNSN1 -= 12
    Cnt['rNSN1'] = rNSN1
    # apply the new ring subset to axial LUTs
    raxLUT = nipet.mmraux.axial_lut(Cnt)
    # michelogram for reduced rings in span-1
    Msn1_c = raxLUT['Msn1']
    # michelogram for full ring case in span-1
    Msn1 = np.copy(axLUT['Msn1'])
    # from full span-1 sinogram index to reduced rings sinogram index
    rlut = np.zeros(rNSN1, dtype=np.int16)
    rlut[Msn1_c[Msn1_c >= 0]] = Msn1[Msn1_c >= 0]
    raxLUT['rLUT'] = rlut

    return raxLUT


def load_data(folder_data, span=1, rings=(0, 64), time=None):
    if time is None:
        time = (0, 0)
        s_time = ''
    else:
        s_time = '_time{}-{}'.format(time[0], time[1])

    filename_data = '{}/data_rings{}-{}_span{}{}.npy'.format(
            folder_data, rings[0], rings[1], span, s_time)

    if not os.path.exists(filename_data):
        print('file {} does NOT exists. Compute it.'.format(filename_data))

        filename_nipet = '{}/data_nipet_span{}{}.npy'.format(folder_data, span,
                                                             s_time)
        if not os.path.exists(filename_nipet):
            print('file {} does NOT exists. Compute it.'
                  .format(filename_nipet))

            # get all the default constants and LUTs
            mMRparams = nipet.mmraux.mMR_params()
            mMRparams['Cnt']['SPN'] = span
            mMRparams['Cnt']['VERBOSE'] = True  # Switch ON verbose mode

            Cnt = mMRparams['Cnt']
            txLUT = mMRparams['txLUT']
            axLUT = mMRparams['axLUT']

            datain = nipet.mmraux.explore_input(folder_data, mMRparams)

            muhdic = nipet.img.mmrimg.hdw_mumap(datain, [1, 2, 4], mMRparams,
                                                use_stored=True)

            muodic = nipet.img.mmrimg.obj_mumap(datain, mMRparams, store=True)

            outpath = os.path.join(datain['corepath'], 'output')

            recpet = nipet.img.rec.osem(datain,
                                        mMRparams,
                                        mu_h=muhdic,
                                        mu_o=muodic, # or mupdct
                                        frames=['fluid', [time[0],time[1]]], # definition of time frames
                                        outpath=outpath,     # output path for results
                                        itr=4,          # number of OSEM iterations
                                        fwhm=0.,        # Gaussian Smoothing FWHM
                                        recmod=3,    # reconstruction mode: -1: undefined, chosen automatically. 3: attenuation and scatter correction, 1: attenuation correction only, 0: no correction (randoms only)
                                        fcomment='',    # text comment used in the file name of generated image files
                                        store_img=True)

            # histogram data in span-1
            hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, t0=time[0],
                                        t1=time[1])
            # get randoms in span-1
            rsn, _ = nipet.lm.mmrhist.rand(hst['fansums'], txLUT, axLUT, Cnt)

            # get norm sino
            nsn = nipet.mmrnorm.get_sino(datain, hst, axLUT, txLUT, Cnt)

            # get scatter in span-1
            ssn, _, _ = nipet.sct.mmrsct.scatter([muhdic['im'], muodic['im']],
                                                 recpet['img'], datain, hst,
                                                 rsn, txLUT, axLUT, Cnt)

            # get registered mri image
            mri_folder = datain['corepath'] + '/T1/mr2pet/'
            if not os.path.exists(mri_folder):
                nipet.img.mmrimg.mr2petAffine(datain, Cnt,
                                              recpet['recon'].fpet)

            mri_list = os.listdir(mri_folder)
            for mri_file in mri_list:
                if mri_file.endswith('nii.gz'):
                    mri_t1 = nipet.img.mmrimg.getnii(mri_folder + mri_file)

            # take the subsets of all the sinograms (scatter sino will be
            # separately calculated)
            psn = hst['psino']
            # also the reconstructed image for reference
            imp = nipet.img.mmrimg.convert2dev(recpet['img'], Cnt)
            # mri image for reference or anatomical priors
            mri = nipet.img.mmrimg.convert2dev(mri_t1, Cnt)

            muo = nipet.img.mmrimg.convert2dev(muodic['im'], Cnt)
            muh = nipet.img.mmrimg.convert2dev(muhdic['im'], Cnt)

            op = operator_mmr(span=span)
            image_attenuation = muo + muh

            att = op.putgaps(op.project_attenuation(image_attenuation))

            # form dictionary of reduced input sinograms
            data_dict = {'psn': psn, 'nsn': nsn, 'rsn': rsn, 'ssn': ssn,
                         'muh': muh, 'muo': muo, 'imp': imp, 'att': att,
                         'mri': mri}

            np.save(filename_nipet, data_dict)

        else:
            print('file {} exists. Load it.'.format(filename_nipet))
            data_dict = np.load(filename_nipet).tolist()

        Cnt, txLUT, axLUT = get_geometry_mmr(span, rings)

        # get precomputed values
        psn = data_dict['psn']
        nsn = data_dict['nsn']
        rsn = data_dict['rsn']
        ssn = data_dict['ssn']
        imp = data_dict['imp']
        att = data_dict['att']
        mri = data_dict['mri']
        muo = data_dict['muo']
        muh = data_dict['muh']
        data_dict = None

        nrings = rings[1] - rings[0]
        if nrings < 64:
            # reduce axial FOV: get updated axial LUTs with new entries also in Cnt
            # (number of sinograms, rings and axial voxels)
            axLUT = reduce_rings(axLUT, Cnt, rings)

            # take the subsets of all the sinograms (scatter sino will be
            # separately calculated)
            psn = psn[axLUT['rLUT'], :, :]
            nsn = nsn[axLUT['rLUT'], :, :]
            rsn = rsn[axLUT['rLUT'], :, :]
            ssn = ssn[axLUT['rLUT'], :, :]
            att = att[axLUT['rLUT'], :, :]

        # also the reconstructed image for reference
        image = imp[:, :, 2*Cnt['RNG_STRT']:2*Cnt['RNG_END']-1]
        image_mr = mri[:, :, 2*Cnt['RNG_STRT']:2*Cnt['RNG_END']-1]

        image_ct = muo[:, :, 2*Cnt['RNG_STRT']:2*Cnt['RNG_END']-1]

        op = operator_mmr(span=span, rings=rings)

        data = np.uint16(op.remgaps(psn))
        psn = None

        background = op.remgaps(rsn + ssn)
        factors = op.remgaps(att * nsn)

        normalisation = op.remgaps(nsn)
        attenuation = op.remgaps(att)
        scatter = op.remgaps(ssn)
        randoms = op.remgaps(rsn)
        rsn, ssn, nsn, att = [None, ] * 4

        np.save(filename_data, (data, background, factors, image, image_mr,
                                image_ct))

        Y = op.range
        x = Y.one()
        sens1 = op.adjoint(x).asarray()
        opn = operator_mmr(span=span, rings=rings, factors=normalisation)
        sens2 = opn.adjoint(x).asarray()
        opf = operator_mmr(span=span, rings=rings, factors=factors)
        sens3 = opf.adjoint(x).asarray()
        x = None

        def show_sino(x, title):
            fig.append(plt.figure())
            clim = [x.min(), x.max()]
            x = Y.element(x)
            misc.imagesc3(op.putgaps(x), clim=clim, title=title)

        def show_image(x, title):
            fig.append(plt.figure())
            misc.imagesc3(x, [x.min(), x.max()], title=title)

        fig = []
        show_sino(normalisation, 'normalisation')
        show_sino(attenuation, 'attenuation')
        show_sino(factors, 'multiplicative correction factors')
        show_sino(data, 'data')
        show_sino(scatter, 'scatter')
        show_sino(randoms, 'randoms')

        show_image(sens1, title='sensitivity geometry')
        show_image(sens2, title='sensitivity geometry + norm')
        show_image(sens3, title='sensitivity geometry + norm + att')

        show_image(muh, title='hardware attenuation')
        show_image(image_ct, title='object attenuation')

        bg = op.putgaps(Y.element(background))
        r = op.putgaps(Y.element(randoms))
        s = op.putgaps(Y.element(scatter))
        d = op.putgaps(Y.element(data))
        a = op.putgaps(Y.element(attenuation))
        n = op.putgaps(Y.element(normalisation))
        f = op.putgaps(Y.element(factors))

        fig.append(plt.figure())
        ind = np.int32(np.array(bg.shape) / 3)
        plt.plot(d[ind[0], ind[1], :], label='data')
        plt.plot(bg[ind[0], ind[1], :], label='background')
        plt.plot(s[ind[0], ind[1], :], label='scatter')
        plt.plot(r[ind[0], ind[1], :], label='randoms')
        plt.legend()
        plt.title('data, 1d')

        fig.append(plt.figure())
        ind = np.int32(np.array(bg.shape) / 2)
        plt.plot(d[ind[0], ind[1], :], label='data')
        plt.plot(10 * f[ind[0], ind[1], :], label='10 x factors')
        plt.plot(10 * a[ind[0], ind[1], :], label='10 x attenuation')
        plt.plot(10 * n[ind[0], ind[1], :], label='10 x normalization')
        plt.legend()
        plt.title('data, 1d')

        fig.append(plt.figure())
        ind = np.int32(np.array(bg.shape) / 1.5)
        plt.plot(d[ind[0], ind[1], :], label='data')
        plt.plot(bg[ind[0], ind[1], :], label='background')
        plt.plot(10 * f[ind[0], ind[1], :], label='10 x factors')
        plt.legend()
        plt.title('data, 1d')
        bg, d, a, n, r, s = [None, ] * 6

        fig.append(plt.figure())
        maxclim = 0.9 * image.max()
        misc.imagesc3(image, clim=[0, maxclim], title='PET recon')

        fig.append(plt.figure())
        maxclim = 0.9 * image_mr.max()
        misc.imagesc3(image_mr, clim=[0, maxclim], title='MRI')

        for i, f in enumerate(fig):
            filename = '{}_{}.png'.format(filename_data[:-4], i)
            f.savefig(filename, bbox_inches='tight')

    else:
        print('file {} exists. Load it.'.format(filename_data))
        data, background, factors, image, image_mr, image_ct = \
            np.load(filename_data)

    return data, background, factors, image, image_mr, image_ct
