# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:41:55 2015

Python implemented MATLAB utilities

@author: Edward
"""

import numpy as np
import re

def getfield(struct, *args): # layered /serial indexing
    """Get value from a field from a dictionary /structure"""
    try:
        for m in args:
            struct = struct[m]
        return(struct)
    except:
        return(None)

def ind2sub(ind, size, order='C'):
    """MATLAB's ind2usb
    order: in 'C' order by default"""
    return(np.unravel_index(ind, size, order=order))


def sub2ind(sub, size, order='C'):
    """MATLAB's sub2ind
    order: in 'C' order by default"""
    return(np.ravel_multi_index(sub, dims=size, order=order))


def getconsecutiveindex(t, N=1):
    """Given a sorted array of integers, find the start and the end of
    consecutive blocks
    E.g. t = [-1, 1,2,3,4,5, 7, 9,10,11,12,13, 15],
    return [1,5; 7,11]
    t: the sorted array of integers
    N: filter for at least N consecutive. Default 1
    """
    x = np.diff(t) == 1
    f = np.where(np.concatenate(([False], x))!=np.concatenate((x, [False])))[0]
    f = np.reshape(f, (2, -1))
    # filter for at least N consecutvie
    f = f[np.diff(f, n=1, axis=1).T[0] > N, :]
    return(f)

def consecutivenum2str(t, N=1):
    """Given a sorted array of integers, return the shortened list
    E.g.
    E.g. t = [-1, 1,2,3,4,5, 7, 9,10,11,12,13, 15],
    return '-1, 1-5, 7, 9-13, 15'
    t: the sorted array of integers
    N: filter for at least N consecutive. Default 1
    """
    f = getconsecutiveindex(t, N=N)
    # blocks
    b = [str(t[m[0]])+'-'+str(t[m[1]]) for m in f]
    # singles
    if N<2:
        s = np.array([], dtype=np.int32)
        for m in f:
            s = np.concatenate((s, np.arange(m[0], m[1]+1, dtype=np.int32)))
        s = np.setdiff1d(np.arange(0, len(t)), s)
        for n in s:
            b.append(str(t[n]))
        f = np.argsort(np.concatenate((f[:,0], s)))
        b = [b[k] for k in f]

    return(', '.join(b))

def str2numeric(lit):
        """"Handling only single numbers"""
        # Handle '0'
        if lit == '0': return 0
        # Hex/Binary
        litneg = lit[1:] if lit[0] == '-' else lit
        if litneg[0] == '0':
            if litneg[1] in 'xX':
                return int(lit,16)
            elif litneg[1] in 'bB':
                return int(lit,2)
            else:
                try:
                    return int(lit,8)
                except ValueError:
                    pass

        # Int/Float/Complex
        try:
            return int(lit)
        except ValueError:
            pass
        try:
            return float(lit)
        except ValueError:
            pass
        return complex(lit)

def str2num(lit):
    """MATLAB behavior of str2num.
    str2num('1') --> 1
    str2num('[5,3,2]') --> [5,3,2]
    Cannot handle matrix yet.
    """
    # Identify all numbers
    lit = re.findall(r"[-+]?\d*\.\d+|\d+", lit)
    # Convert to a list of numbers
    lit = [str2numeric(a) for a in lit]
    lit = lit[0] if len(lit)==1 else lit
    return(lit)


def findpeaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like indices of the peaks in `x`.
    pks: height of detected peaks.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = findpeaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/
                                    notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from findpeaks import findpeaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = findpeaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> findpeaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> findpeaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> findpeaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> findpeaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> findpeaks(x, threshold = 2, show=True)

    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    pks = np.array([x[p] for p in ind])

    return(ind, pks)

def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return(m_i)

def isempty(m):
    """Return true if:
    a). an empty string
    b). an list of length zero
    c). an tuple of length zero
    d). an numpy array of length zero
    """
    if isinstance(m, (list, tuple, str, np.ndarray)):
        return(len(m) == 0)

def rms(x):
    """Root mean square of an array"""
    return(np.sqrt(np.mean(x**2)))


def is_numeric(obj): # to check if a numpy object is numeric
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

    if not is_numeric(A) or np.ndim(A)!=2 or any([s!=4 for s in np.shape(A)]):
        raise(IOError(\
        "Order expression '%s' did not return a valid 4x4 matrix."%(order)))

    return(A, T, R, Z, S)


def padzeros(x):
    """Pad zeros to make the array length 2^n for fft or filtering
    y, l = padzeros(x)

    x: input vector
    y: zero-padded vector
    l: length of the original array
    """
    l = len(x)
    pad = 2**nextpow2(l)
    if (pad - l) < (0.1*l):
        pad = 2**(nextpow2(l)+1)
    pad = int(pad - l) # legnth of padding
    x = np.concatenate((x, np.zeros(pad)))
    return(x, l)

def longest_repeated_substring(lst, ignore_nonword=True, inall=True):
    """given a list of strings, find common substrings. Example:
    ['Neocortex A', 'Neocortex B', 'Neocortex C'], aligning to the left, yields
    'Neocortex'.

    * ignore_nonword: By default ignore non-word characters, and only look for
          characters in [a-zA-Z0-9_]. To include everything, set this to False.
          Can also specify a set of characters to remove in regular expression.

    * inall: does the longest string has to be in all strings in the list.
        True: longest string has to be in all strings. Default.
        False: at least in 2 strings
        Some integer N: at least in N string
    """
    longest = None
    if isinstance(inall, bool):
        count = len(lst)-1 if inall else 1
    else:
        count = inall

    # Look for the word
    for word in lst:
        for i in range(len(word)):
            for j in range(i+1, len(word)+1):
                if ((longest is None or (j - i > len(longest))) and
                    sum(word[i:j] in w for w in lst) > count):
                    longest = word[i:j]

    # Remove non-word character depending on the option
    if ignore_nonword:
        if isinstance(ignore_nonword, bool):
            ignore_nonword = '[^a-zA-Z0-9_]'
        longest = re.sub(ignore_nonword, '', longest)

    return(longest)


def sort_nicely( l ):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l
