"""Stochastic Primal-Dual Hybrid Gradient (SPDHG) algorithms"""

from __future__ import print_function, division
import numpy as np
import odl

def pdhg(x, f, g, A, tau, sigma, niter, **kwargs):
    """Computes a saddle point with PDHG.

    This algorithm is the same as "algorithm 1" in [CP2011a] but with
    extrapolation on the dual variable.


    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : function
        Functional Y -> IR_infty that has a convex conjugate with a
        proximal operator, i.e. f.convex_conj.proximal(sigma) : Y -> Y.
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : function
        Operator A : X -> Y that possesses an adjoint: A.adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    callback : callable
        Function called with the current iterate after each iteration.

    References
    ----------
    [CP2011a] Chambolle, A and Pock, T. *A First-Order
    Primal-Dual Algorithm for Convex Problems with Applications to
    Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011),
    pp 120-145.
    """

    def fun_select(k):
        return [0]

    prob = [1]

    f = odl.solvers.SeparableSum(f)
    A = odl.BroadcastOperator(A, 1)

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y_tmp = A.range.zero()
    else:
        y_tmp = A.range.element([y])

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None:
        if y_tmp.norm() == 0:
            z = A.domain.zero()
        else:
            z = A.adjoint(y_tmp)

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    spdhg_generic(x, y_tmp, z, f, g, A, tau, [sigma], niter, prob, fun_select,
                  callback)

    if y is not None:
        y.assign(y_tmp[0])


def spdhg(x, f, g, A, tau, sigma, niter, **kwargs):
    """Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y : dual variable
        Dual variable is part of a product space. By default equals 0.
    z : variable
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    prob: list
        List of probabilities that an index i is selected each iteration. By
        default this is uniform serial sampling, p_i = 1/n.
    callback : callable
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] M. J. Ehrhardt, P. J. Markiewicz, P. Richtarik, J. Schott,
    A. Chambolle and C.-B. Schoenlieb, *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """

    # Probabilities
    prob = kwargs.pop('prob', None)
    if prob is None:
        prob = [1 / len(A)] * len(A)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=prob))]

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None:
        if y.norm() == 0:
            z = A.domain.zero()
        else:
            z = A.adjoint(y)

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    spdhg_generic(x, y, z, f, g, A, tau, sigma, niter, prob, fun_select,
                  callback)


def spdhg_generic(x, y, z, f, g, A, tau, sigma, niter, prob, fun_select,
                  callback):
    """Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    y : dual variable
        Dual variable is part of a product space. By default equals 0.
    z : variable
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations
    callback : callable
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] M. J. Ehrhardt, P. J. Markiewicz, P. Richtarik, J. Schott,
    A. Chambolle and C.-B. Schoenlieb, *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """

    # Initialize variables
    z_bar = z.copy()
    dz = z.copy()
    y_old = y.copy()

    # Save proximal operators
    prox_f = [fi.convex_conj.proximal(si) for fi, si in zip(f, sigma)]
    prox_g = g.proximal(tau)

    # run the iterations
    for k in range(niter):

        # select block
        selected = fun_select(k)

        # update dual variable and z, z_relax
        z_bar.assign(z)
        for i in selected:

            # update dual variable
            # y = prox(y + sigma * Ax)
            y_old[i].assign(y[i])
            A[i](x, out=y[i])
            y[i] *= sigma[i]
            y[i] += y_old[i]
            prox_f[i](y[i], out=y[i])

            # update adjoint of dual variable
            # dz = A*(y - y_old)
            y_old[i] -= y[i]
            y_old[i] *= -1
            A[i].adjoint(y_old[i], out=dz)
            z += dz

            # compute extrapolation
            # z_bar = z + (1 + 1 / p) * dz
            dz *= 1 + 1 / prob[i]
            z_bar += dz

        # update primal variable
        # x = prox(x - tau * z_bar)
        z_bar *= tau
        x -= z_bar
        prox_g(x, out=x)

        if callback is not None:
            callback([x, y])