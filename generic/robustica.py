# -*- coding: utf-8 -*-
"""
RobustICA Python implementation
Adapted from original author's MATLAB script

Created on Mon Dec 21 19:24:57 2015
@author: Edward
"""

import numpy as np

def kurt_gradient_optstep(w, X, s, P, wreal, verbose=False):
    """ Computes optimal step size in the gradient-based optimization of the
        normalized kurtosis contrast (single iteration).

    Data-based version.

    See references below for details.

    SYNTAX:
            g, mu_opt, norm_g = kurt_gradient_optstep(w, X, s, P, wreal)

    INPUTS:
             w      : current extracting vector coefficients

             X      : sensor-output data matrix (one signal per row, one sample
                      per column)

             s      : source kurtosis sign; if zero, the maximum absolute value
                      of the contrast is sought

             P      : projection matrix (used in deflationary orthogonalization;
                      identity matrix otherwise)

             wreal  : if different from zero, keep extracting vector real valued
                      by retaining only the real part of the gradient (useful,
                      for instance, in separating real-valued mixtures in the
                      frequency domain, as in the RobustICA-f algorithm).

             verbose:  verbose operation if true
                       * default: False (quiet operation).

    OUTPUTS:
             g      : search direction (normalized gradient vector)

             mu_opt : optimal step size globally optimizing the normalized
                      kurtosis contrast function along direction g from f

             norm_g : non-normalized gradient vector norm.

    REFERENCES:

    - V. Zarzoso and P. Comon, <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/tnn10.pdf">"Robust independent component analysis by iterative maximization</a>
      <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/tnn10.pdf">of the kurtosis contrast with algebraic optimal step size"</a>,
      IEEE Transactions on Neural Networks, vol. 21, no. 2, pp. 248-261, Feb. 2010.

    - V. Zarzoso, P. Comon and M. Kallel,  <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/eusipco06.pdf">"How fast is FastICA?"</a>,
      in: Proceedings EUSIPCO-2006, XIV European Signal Processing Conference,
      Florence, Italy, September 4-8, 2006.

    Please, report any bugs, comments or suggestions to
    <a href = "mailto:zarzoso@i3s.unice.fr">zarzoso(a)i3s.unice.fr</a>.

    HISTORY:

        <modification date>: - <modification details>

    -- 2014/11/21: Version 3 release ------------------------------------------

        2014/06/25: - added 'wreal' input parameter to allow the separation of
                      real-valued mixtures in complex (e.g., frequency) domain

     -- 2010/02/16: Version 2 release -----------------------------------------

        2010/02/09: - include gradient norm as output parameter (for use as an
                      additional termination criterion)

        2009/03/04: - removed test for constant contrast; sometimes the
                      algorithm stopped too early, because the contrast was
                      not actually constant, leading to suboptimal extraction
                      results
                    - if best candidate root is complex valued, its real part
                      can be retained as optimal step size, but contrast is not
                      guaranteed to increase monotonically in that case; to
                      avoid this problem, only the real parts of the roots
                      are considered

        2009/03/02: - simplified expressions of gradient and optimal step size,
                      as in TNN paper

        2008/04/01: - problem encountered on 2008/03/25 using orthogonalization:
                      due to nearly-zero gradient appearing when just one
                      source is left, since the contrast function then becomes
                      constant; normalization after projection in such a case
                      destroys co-linearity between gradient and extracting
                      vector (checking for zero gradient should probably be
                      used as additional termination test in the next version
                      of the algorithm; see modification on 2010/02/09)

     -- 2008/03/31: Version 1 release ----------------------------------------------------------------

        2008/03/26: - added this help

        2008/03/25: - projecting the gradient after normalization seems to
                      improve conditioning and accelerate convergence in the
                      extraction of the last sources

        2008/03/24: - created by Vicente Zarzoso
                      (University of Nice - Sophia Antipolis, France).
    """

    # verbose = 0
    L, T = np.shape(X)

    mu_opt = 0  # default optimal step-size value
    norm_g = 0 # initialize gradient norm

    # Compute search direction (gradient vector)

    # compute necessary interim values
    y = w.conj().T * X

    ya2 = y * y.conj()
    y2 = y * y
    ya4 = ya2 * ya2

    Eya2 = np.mean(ya2)
    Ey2 = np.mean(y2)
    Eya4 = np.mean(ya4)

    if abs(Eya2) < np.finfo(float).eps: # check for zero denominator
        if verbose:
            print('>>> OPT STEP SIZE: zero power\n')
        g = np.zeros(L)
        norm_g = 0
        return(g, mu_opt, norm_g)

    # compute gradient if contrast denominator is not null
    Eycx = X*y.conj().T/T
    Eyx = X*y.T/T
    Ey3x = X*(ya2*y).conj().T/T

    # contrast numerator and denominator at current point
    p1 = Eya4 - abs(Ey2)^2
    p2 = Eya2

    g = 4.*( (Ey3x - Eyx*Ey2.conj().T)*p2 - p1*Eycx )/p2**3.;

    g = P*g           # project if required (normalize later)

    norm_g = np.linalg.norm(g)

    if norm_g < np.finfo(float).eps: # check for zero
        if verbose:
            print('>>> OPT STEP SIZE: zero gradient\n')
        return(g, mu_opt, norm_g)

    # keep only real part if real-valued extracting vectors are required
    if wreal:
        g = np.real(g)

    # normalize the gradient -> parameter of interest: direction improves
    # conditioning of opt step-size polynomial
    g = g / norm_g

    # Compute optimal step size
    gg = g.conj().T * X

    # calculate interim values for contrast rational function
    ya2 = y * y.conj()
    ga2 = gg * gg.conj()
    reygc = (y * gg.conj()).real
    g2 = gg * gg
    yg = y * gg

    Eya2reygc = np.mean(ya2*reygc)
    Ereygc2 = np.mean(reygc**2.)
    Ega2reygc = np.mean(ga2*reygc)
    Ega4 = np.mean(ga2**2.)
    Eya2ga2 = np.mean(ya2*ga2)
    Ega2 = np.mean(ga2)
    Ereygc = np.mean(reygc)
    Eg2 = np.mean(g2)
    Eyg = np.mean(yg)

    h0 = Eya4 - abs(Ey2)**2
    h1 = 4*Eya2reygc - 4*real(Ey2*Eyg.conj().T)
    h2 = 4*Ereygc2 + 2*Eya2ga2 - 4*abs(Eyg)^2 - 2*real(Ey2*Eg2.conj().T)
    h3 = 4*Ega2reygc - 4*real(Eg2*Eyg.conj().T)
    h4 = Ega4 - abs(Eg2)^2

    P = [h4, h3, h2, h1, h0]

    i0 = Eya2
    i1 = 2*Ereygc
    i2 = Ega2

    Q = [i2, i1, i0]

    # normalized kurtosis contrast = P/Q^2 - 2

    a0 = -2*h0*i1 + h1*i0
    a1 = -4*h0*i2 - h1*i1 + 2*h2*i0
    a2 = -3*h1*i2 + 3*h3*i0
    a3 = -2*h2*i2 + h3*i1 + 4*h4*i0
    a4 = -h3*i2 + 2*h4*i1

    p = [a4, a3, a2, a1, a0]

    # normalized kurtosis contrast derivative = p/Q^3
    # ALTERNATIVE METHOD to compute optimal step-size polynomial oefficients
    #
    # # obtain contrast-function polynomials
    #
    # p11 = [Ega4, 4*Ega2reygc, 4*Ereygc2+2*Eya2ga2, 4*Eya2reygc, Eya4];
    # p13 = [Eg2, 2*Eyg, Ey2];
    # P = p11 - conv(p13, conj(p13));     # numerator
    # Q = [Ega2, 2*Ereygc, Eya2];         # square-root of denominator
    #
    # # compute derivatives
    # Pd = [4, 3, 2, 1].*P(1:4);
    # Qd = [2, 1].*Q(1:2);
    #
    # # contrast derivative numerator
    # p = conv(Pd, Q) - 2*conv(Qd, P);

    rr = np.roots(p).real        # keep real parts only

    Pval = np.polyval(P, rr)
    Q2val = np.polyval(Q, rr)**2

    # check roots not shared by denominator
    nonzero_Q2val = np.where(Q2val > np.finfo(float).eps)[0]
    # Note: in theory, the denominator can never cancel out if the gradient is
    # used as a search direction, due to the orthogonality between the
    # extracting vector and the corresponding gradient (only exception: if it
    # is the last source to be extracted; but this scenario is detected by the
    # gradient norm)

    if len(nonzero_Q2val) == 0:
        if verbose:
            print('>>> OPT STEP SIZE: all roots shared by denominator\n')
            print('Pval = ')
            print(Pval.conj().T)
            print('\nQ2val = ')
            print(Q2val.conj().T)
            print('\np = ')
            print(p)
            print('\nP = ')
            print(P)
            print('\nQ = ')
            print(Q)
            Q2 = np.convolve(Q, Q)
            P_Q2 = P/Q2
            print('\nP_Q2 = ')
            print(P_Q2)
            print('\n')
        return(g, mu_opt, norm_g)

    Pval = Pval[nonzero_Q2val]
    Q2val = Q2val[nonzero_Q2val]
    rr = rr[nonzero_Q2val]

    Jkm_val = Pval / Q2val - 2.    # normalized kurtosis

    if s:
        Jkm_val = (s*Jkm_val).real # maximize or minimize kurtosis value,
                                  # depending on kurtosis sign
    else:
        Jkm_val = abs(Jkm_val)   # maximize absolute kurtosis value,
                                # if no sign is given
    im = np.argmax(Jkm_val)
    mu_opt = rr[im]            # optimal step size

    return(g, mu_opt, norm_g)


def deflation_regression(X, s, dimred):
    """ Performs deflation by subtracting the estimated source contribution to
        the observations as:

        X' = X - h*s

    The source direction h is estimated via the least squares solution to the
    linear regression problem:

        h_opt = arg min_h ||X - h*s||^2 = X*s'/(s*s').

    SYNTAX:
            Xn = deflation_regression(X, s, dimred)


    INPUTS:
             X      : observed data (one signal per row, one sample per column)

             s      : estimated source (row vector with one sample per column)

             dimred : perform dimensionality reduction if parameter different
                      from zero.

    OUTPUT:
             Xn     : observed data after subtraction (one signal per row,
                      one sample per column).

    Please, report any bugs, comments or suggestions to
    <a href = "mailto:zarzoso@i3s.unice.fr">zarzoso(a)i3s.unice.fr</a>.


    HISTORY:

        <modification date>: - <modification details>

    -- 2014/11/21: Version 3 release ------------------------------------------

    -- 2010/02/16: Version 2 release ------------------------------------------

    -- 2008/03/31: Version 1 release ----------------------------------------

       2008/03/26: - added this help

       2008/03/18: - if required, perform dimensionality reduction via QR
                     decomposition

       2008/03/13: - created by Vicente Zarzoso
                     (I3S Laboratory, University of Nice Sophia Antipolis,
                     CNRS, France).
    """
    s2 = s * s.conj().T # extracted source power times sample size

    if abs(s2) < np.finfo(float).eps:
        # don't perform subtraction if estimated component is null
        return(X)

    h = X * s.conj().T / s2  # source direction estimated via least squares

    if dimred:
        # with dimensionality reduction (old version)  ***********************
        n = len(h)
        Q = np.concatenate((h, np.eye(n, n-1), axis=0)
        Q, R = np.linalg.qr(Q)
        Q = Q[:,1:n]   # orthonormal basis of orhogonal subspace of h
        X = Q.conj().T * X #remaining contribution with dimensionality reduction
    else:
        # without dimensionality reduction
        X = X - h*s

        # if dimred   ### an alternative version?  *** TO BE TESTED ***
        #     [n, T] = size(X)
        #     [V, S, U] = svd(X', 0) #'economy' SVD
        #     hU = abs(h'*U)
        #     diagS = diag(S)
        #     X = sqrt(T)*V(:, 1:n-1)'
        #     pause
        # end # if dimred

    return(X)

def robsutica(X, deftype='orthogonalization', dimred=False, kurtsign=0.,
              maxiter=1000, prewhi=True, tol=1E-3, verbose=False, wini=None,
              wreal=False):
    """ Kurtosis-based RobustICA method for deflationary ICA/BSS
        (see references below for details).

    SYNTAX:
            S, H, niter, W = robustica(X, **kwargs)

    INPUTS:
             X       : observed signals (one row per signal, one column per
                       sample)

             deftype:  deflation type: 'orthogonalization', 'regression'
                        * default: 'orthogonalization'

             dimred:   dimensionality reduction in regression if parameter
                       different from zero; (not used in deflationary
                       orthogonalization)
                        * default: false

             kurtsign: source kurtosis signs (one element per source);
                        maximize absolute normalized kurtosis if corresponding
                        element = 0;
                        * default: zero vector (maximize absolute normalized
                          kurtosis for all sources)

             maxiter:  maximum number of iterations per extracted source;
                        * default: 1000

             prewhi:   prewhitening (via SVD of the observed data matrix);
                        * default: true

             tol:      threshold for statistically-significant termination
                             test of the type
                             ||wn - p*w||/||w|| < tol/sqrt(sample size);
                             (up to a phase shift p)
                       termination is also tested by comparing the gradient
                             norm according to:
                             ||g|| < tol/sqrt(sample size);
                       termination test is not used if tol < 0, so that the
                             algorithm runs the maximu number of iterations
                             (except if optimal step size reaches a null value)
                        * default: 1E-3

             verbose:  verbose operation if true
                        * default: False (quiet operation).

             wini:     extracting vectors initialization for RobustICA
                       iterative search; if empty, identity matrix of suitable
                       dimensions is used
                        * default: None

             wreal:    if different from zero, keep extracting vector real
                       valued by retaining only the real part of the gradient;
                       useful, for instance, in separating real-valued mixtures
                       in the frequency domain, as in the RobustICA-f algorithm
                        * default: False.

    OUTPUTS:
             S       : estimated sources signals (one row per signal,
                       one column per sample)

             H       : estimated mixing matrix

             niter    : number of iterations (one element per extracted source)

             W       : estimated extracting vectors
                      (acting on whitened observations if prewhitened is
                      required; otherwise, acting on given observations).

    EXAMPLES:

    >>  S = robustica(X, **kwargs);

    - RobustICA with prewhitening, deflationary orthogonalization, identity
      matrix initialization, up to 1000 iteratons per source, termination
      threshold 1E-3/(sample size), without aiming at any specific source
      (default)

          S = robustica(X)

    - RobustICA with prewhitening and regression-based deflation:

          ... deftype='regression' ...

    - RobustICA without prewhitening, with regression-based deflation:

          ... deftype='regression', prewhi=False ...

    - RobustICA without prewhitening, with regression-based deflation and
      random initialization:

          ... deftype='regression', prewhi=False, \
                    wini=np.randn(np.shape(X)[0]) ....

    - RobustICA without prewhitening, with regression-based deflation and
      dimensionality reduction:

          ... deftype='regression', dimred=True, prewhi=False ...

    - RobustICA with prewhitening, regression-based deflation, verbose operation:

          ... deftype='regression', verbose=True ...

    - RobustICA with prewhitening, deflationary orthogonalization, and exactly
      10 iterations per independent component:

          ... tol=-1, maxiter=10 ...

    - RobustICA with prewhitening, deflationary orthogonalization, targeting
      first the sub-Gaussian and then the super-Gaussian sources in a square
      mixture of 5 sub-Gaussian and 5 super-Gaussian sources:

          ... kurtsign=[np.ones(1,5), -np.ones(1,5)] ...

    - RobustICA with prewhitening, regression-based deflation, targeting first
      a sub-Gaussian source:

          ... deftype=regression', \
                    kurtsign=np.insert(np.zeros(np.shape(X)[0]-1), 0, -1) ...


    REFERENCES:

    - V. Zarzoso and P. Comon, <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/tnn10.pdf">"Robust Independent Component Analysis by Iterative Maximization</a>
      <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/tnn10.pdf">of the Kurtosis Contrast with Algebraic Optimal Step Size"</a>,
      IEEE Transactions on Neural Networks, vol. 21, no. 2, pp. 248-261,
      Feb. 2010.

    - V. Zarzoso and P. Comon, <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/ica07.pdf">"Comparative Speed Analysis of FastICA"</a>,
      in: Proceedings ICA-2007, 7th International Conference on Independent Component Analysis
      and Signal Separation, London, UK, September 9-12, 2007, pp. 293-300.

    - V. Zarzoso, P. Comon and M. Kallel,  <a href = "http://www.i3s.unice.fr/~zarzoso/biblio/eusipco06.pdf">"How Fast is FastICA?"</a>,
      in: Proceedings EUSIPCO-2006, XIV European Signal Processing Conference,
      Florence, Italy, September 4-8, 2006.


    Please, report any bugs, comments or suggestions to
    <a href = "mailto:zarzoso@i3s.unice.fr">zarzoso(a)i3s.unice.fr</a>.


    HISTORY:

        <modification date>: - <modification details>

    -- 2014/11/21: Version 3 release ------------------------------------------

       2014/06/25: - added 'wreal' input parameter to allow the separation of
                      real-valued mixtures in a complex domain (e.g., after
                      Fourier transform)
                    - simplified calling syntax using cell-array input argument
                      (irrelevant for Python)
    -- 2010/02/16: Version 2 release ------------------------------------------

       2010/02/09: - added termination test based on gradient norm

       2009/03/02: - project extracting vector before normalization

       2009/02/02: - variable 'thmu' (for step-size based termination test)
                     removed, as it was not used

    -- 2008/03/31: Version 1 release ------------------------------------------

       2008/12/03: - modified help info about output parameter W
                     (extracting vectors act on whitened observation if
                     prewhitening is required)

       2008/03/26: - added this help

       2008/03/13: - created by Vicente Zarzoso
                     (I3S Laboratory, University of Nice Sophia Antipolis,
                     CNRS, France).
    """

    # record size of the input signal
    n, T = np.shape(X)

    # remove mean from each column
    X = X - np.mean(X, axis=1)[:, np.newaxis]

    # prewhitening if prewhi=True
    if prewhi:
        if verbose:
            print(">>> Prewhitening\n")

        V, D, U = np.linalg.svd(X.conj().T, 0)  # economy SVD of data matrix
        B = U * D / np.sqrt(T)                  # PCA mixing-matrix estimate
        Z = np.sqrt(T) * V.conj().T             # PCA source estimate
    else:
        Z = X

    # RobustICA algorithm
    dimobs = n # number of remaining observationd (may chnage under
               # dimensionality reduction)
    W = np.zeros(n) # extracting vectors
    I = np.eye(n)
    P = I # projection matrix for deflationary orthogonalization (if required)

    tol = tol / np.sqrt(T) # a statistically-significant termination threshold
    tol2 = np.sign(tol) * tol**2 / 2 # the same threshold in terms of
                                 # extracting vectors's absolute scaler product
    niter = np.zeros(n) # number of iterations

    if deftype == 'regression':
        do_reg = True
        type_text = 'regression-based deflation'
        if dimred: # only used in regression mode
            type_text += ' and dimensionality reduction'
        else: # default
            type_text += ', no dimensionality reduction'

    else:
        do_reg = False
        type_text = 'deflationary orthogonalization'

    if verbose:
        print('\n>>>RobustICA with %s\n', type_text)

    # iterate over all sources
    for k in range(n):
        if verbose:
            print('> source # %d :\n', k)

        it = 0
        keep_going = True

        w = wini[:, k] # initialization

        # keep only required number of components
        if do_reg:
            w = w[(n-dimobs):n]

        w = w / np.linalg.norm(w) # normalization
        w = P * w # project onto extracted vectors' orthogonal subspace
                  # (if deflationary orthogonalization)

        signkurt = kurtsign[k] # kurtosis sign of next source to be estimated

        # iterate to extract one source
        while keep_going:
            it += 1

            # compute KM optimal step size for gradient descent
            g, mu_opt, norm_g = kurt_gradient_optstep(w, Z, signkurt, P, wreal,
                                                        verbose=verbose)

            # update extracting vector and project if required
            wn = P * (w + mu_opt * g)

            wn = wn / np.linalg.norm(wn) # normalize

            # extracting vector convergence test
            th = np.abs(1. - np.abs(wn.conj().T*w))

            w = wn

            if th < tol2 or norm_g < tol or it>maxiter or mu_opt == 0:
                # finish when extracting vector converges, the gradient is too
                # small, too many iterations have been run, or the optimal
                # step-size is zero
                keep_going = False
        # end while keep_going

        if do_reg:
            # zero-padding to account for dimensionality reduction in regression
            W[:, k] = np.concatenate((w, np.zeros(n-dimobs, 1)), axis=0)
        else:
            W[:, k] = w # estimated extracting vector

        s = w.conj().T * Z # estimated source
        S[k, :] = s
        niter[k] = it # number of iterations

        if verbose:
            print('%d iterations\n', it)

        if do_reg:
            Z = deflation_regression(Z, s, dimred) # regression + subtraction
            dimobs = np.shape(Z)[0] # recompute observation dimension, just in
                                   # case it has changed during deflation
            P = np.eyeO(dimobs) # P is not required, but its dimensions should
                                # decrease according to Z
        else:
            # projection matrix for orthogonalization (if required)
            P = I - W * W.conj().T
    # end for k

    if verbose:
        print('\nTotal number of iterations: %d\n', np.sum(niter))

    # Mixing matrix estimation
    H = X * S.conj().T * np.linalg.pinv(S * S.conj().T) # LS estimate

    return(S, H, niter, W)


if __name__ == '__main__':
    # test
    pass
