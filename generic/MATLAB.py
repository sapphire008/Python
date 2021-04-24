# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:41:55 2015

Python implemented MATLAB utilities

@author: Edward
"""
import sys
import os
import re
import glob
import operator
from pdb import set_trace
from collections import OrderedDict, Iterable

import numpy as np
import pandas as pd

import scipy as sp
from scipy import stats
from scipy import sparse
from skimage.draw import polygon


def getfield(struct, *args): # layered /serial indexing
    """Get value from a field from a dictionary /structure"""
    try:
        for m in args:
            struct = struct[m]
        return(struct)
    except:
        return(None)

def ind2sub(ind, size, order='C'):
    """MATLAB's ind2sub
    order: in 'C' order by default"""
    return(np.unravel_index(ind, size, order=order))


def sub2ind(sub, size, order='C'):
    """MATLAB's sub2ind
    order: in 'C' order by default"""
    return(np.ravel_multi_index(sub, dims=size, order=order))


def getconsecutiveindex(t, N=1, interval=True):
    """Given a sorted array of integers, find the start and the end of
    consecutive blocks
    E.g. t = [-1, 1,2,3,4,5, 7, 9,10,11,12,13, 15],
    return [1,5; 7,11]
    t: the sorted array of integers
    N: filter for at least N consecutive. Default 1
    interval: if True, we are filtering by N consecutive intervals instead of
    N consecutive numbers

    """
    x = np.diff(t) == 1
    f = np.where(np.concatenate(([False], x))!=np.concatenate((x, [False])))[0]
    f = np.reshape(f, (-1, 2))
    # filter for at least N consecutvie
    f = f[(int(not(interval))+np.diff(f, n=1, axis=1).T[0]) >= N, :]
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
        # Handle 'inf'
        if lit.lower() == '-inf': return -np.inf
        if lit.lower() == 'inf': return np.inf
        # Handle nan
        if lit.lower() == 'nan': return np.nan
        # Handle None
        if lit.lower() == 'none': return None
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
        try:
            complex(lit)
        except ValueError:
            raise(ValueError('String must contain only numerics'))

def str2num(lit, dlimiter=';'):
    """MATLAB behavior of str2num.
    str2num('1') --> 1
    str2num('[5,3,2]') --> [5,3,2]
    str2num('[5,3,2;2,3,1]') --> [[5,3,2],[3,2,1]]
    """
    # Separate the string by semicolon ";" for each row
    lit = lit.split(dlimiter)
    # Identify all numbers
    lit = [re.findall(r"[+-]?\d+(?:\.\d+)?", l) for l in lit]
    # Convert to a list of numbers
    lit = [[str2numeric(a) for a in l] for l in lit]
    lit = lit[0] if len(lit)==1 else lit # squeeze for vectors
    lit = lit[0] if len(lit)==1 else lit # squeeze again for singleton
    return(lit)

def rms(x):
    """Root mean square of an array"""
    return(np.sqrt(np.mean(x**2)))

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
    .. [1] http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
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
        return  np.array([], dtype=int), np.array([], dtype=float)
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

def isempty(m, singleton=True):
    """Return true if:
    a). an empty string
    b). a list of length zero
    c). a tuple of length zero
    d). a numpy array of length zero
    e). a singleton that is not None

    if singleton is true: treat np.ndarray as a single object, and return
    false only when the ndarray is an empty array
    """
    if isinstance(m, (list, tuple, str)):
        if len(m) == 0:
            return True
        else:
            return all([not x for x in m])
    elif isinstance(m, np.ndarray):
        if len(m) == 0:
            return True
        else:
            if singleton:
                return False
            else: # matrix
                K = np.empty_like(m, dtype=np.bool)
                K_shape = np.shape(K)
                for k in range(np.size(K)):
                    ijk = np.unravel_index(k, K_shape, order='C')
                    try:
                       K[ijk] = True if m[ijk].size==0 else False
                    except:
                       K[ijk] = False
                return K
    else:
        return False if m else True

def isnumber(obj):
    """Determine if the object is a single number"""
    attrs = ['__add__', '__sub__', '__mul__', '__pow__', '__abs__']   # need to have
    attrs_neg = ['__len__']  # cannot have
    return (all(hasattr(obj, attr) for attr in attrs)) and (not any(hasattr(obj, attr) for attr in attrs_neg))

def isnumeric(obj):
    """Check if an object is numeric, or that elements in a list of objects
    are numeric. Set all=True to return a signle boolean if all elements of the list
    is numeric"""
    # Allow application to iterables
    f_vec = np.frompyfunc(isnumber, 1, 1)
    tf = f_vec(obj)
    if isinstance(tf, np.ndarray):
        tf = tf.astype(dtype=bool)
    return tf

def isstrnum(obj):
    """Check if a string can be converted into numeric"""
    try:
        str2numeric(obj)
        return True
    except:
        return False

def isrow(v):
    v = np.asarray(v)
    return True if len(v.shape)==1 else False

def iscol(v):
    v = np.asarray(v)
    return True if len(v.shape)==2 and v.shape[1] == 1 else False

def isiterable(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def isvector(v):
    v = np.asarray(v)
    if len(v.shape)==0:
        return False
    elif len(v.shape)==1:
        return True
    else: # len(v.shape) > 1:
        if v.shape[0] == 1 or v.shape[1] == 1:
            return True
        else:
            return False

def ismatrix(v):
    v = np.asarray(v)
    shape = v.shape
    if len(shape) == 2:
        return True if all([s>1 for s in shape]) else False
    elif len(shape) > 2:
        return True if sum([s>1 for s in shape])>=2 else False
    else:
        return False

def listintersect(*args):
    """Find common elements in lists"""
    args = [x for x in args if x is not None] # get rid of None
    def LINT(A,B):  #short for list intersection
        return list(set(A) & set(B))
    if len(args) == 0:
        return(None)
    elif len(args) == 1:
        return(args[0])
    elif len(args) == 2:
        return(LINT(args[0],args[1]))
    else:
        newargs = tuple([LINT(args[0], args[1])]) + args[2:]
        return(listintersect(*newargs))

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

def cell2array(C):
    """Helpful when reading MATLAB .mat files containing cellarray"""
    n, m = C.shape
    K = np.zeros((n,m))
    K = K.tolist()
    for i in range(n):
        for j in range(m):
            tmp = C[i][j]
            if tmp.shape == (1,1):
                tmp = tmp[0][0]
            elif tmp.shape == (1,):
                tmp = tmp[0]
            elif tmp.shape[0] == 1 or tmp.shape[1] == 1:
                tmp = tmp.flatten()
            K[i][j] = tmp
    return K

def cell2df(C):
    """Take a step further than cell2array to convert cell array just read from
    MATLAB's .mat file into a pandas DataFrame
    """
    df = cell2array(C)
    df = pd.DataFrame(df[1:], columns=df[0])
    return df


def dict2df(mydict, colnames=None, addindex=False):
    """ Converting a dictionary into a Pandas data frame. Example:
    df = dict2df(mydict, colnames=['Drug', 'BifTime'], addindex=True)
    Converting
    mydict ={'ChR2': np.asarray([25, 725, 225, 175, 825, 1075, 825, 125, 325, 875, 325, 575, 1325, 725]),
             'Terfenadine': np.asarray([725, 275, 175, 675, 525, 775]),
             'XE991': np.asarray([175, 75, 75, 125, 125]),
             'NS8593': np.asarray([25, 25, 25, 75, 75, 75, 75])}

    into a data frame:

    index           Drug        BifTime
    0               ChR2          25
    1               ChR2          725
    2               ChR2          225
         ...........................
    0            Terfnadine       725
    1            Terfenadine      275
         ...........................
         ...........................
    0              NS8593         25
         ...........................

    colnames: column names of [key, values]
    addindex: add a column called "index" as the first column

    """
    df = pd.DataFrame.from_dict(mydict, orient='index').transpose()
    df['index'] = df.index
    df = pd.melt(df, id_vars=["index"])
    if not addindex:
        df = df.drop(["index"], axis=1)
    # Get rid of NaNs
    try:
        df = df.loc[~np.isnan(df['value'])]
    except:
        pass
    df = df.reset_index(drop=True)
    if colnames is not None:
        df.columns = (["index"] if addindex else [] )+ list(colnames)

    return df

def cell2list_b(C):
    """From loaded MATLAB cell to Python's list
    This assumes each element of the table has only 1 elemtns.
    Legacy: use cell2array for more general purposes.
    """
    n, m = C.shape
    K = np.zeros((n,m))
    K = K.tolist()
    for i in range(n):
        for j in range(m):
            tmp = C[i][j][0]
            if isinstance(tmp, (np.ndarray, list)):
                tmp = tmp[0]

            K[i][j] = tmp
    return K

def list2df(K):
    """From Python's list to panadas' data frame"""
    headers = K[0]
    df = {}
    for n, h in enumerate(headers):
        df[h] = [c[n] for c in K[1:]]

    df = pd.DataFrame(data=df, columns=headers)

    return df


def remap_dict_data_type(mapping, func_key=str, func_val=lambda x: x):
    """Change the data type of keys and values of dictionary"""
    return {func_key(k): func_val(v) for k,v in mapping.items()}

def sort_nicely( l ):
    """ Sort the given list of strings in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def sortrows(A, col=None):
    """Return sorted A, and index, such that A_old[index] = A_new"""
    A = np.asarray(A)
    if not ismatrix(A):
        if isrow(A):
            return np.sort(A), np.argsort(A)
        else:
            return np.sort(A, axis=0), np.argsort(A, axis=0)

    # Sort the whole row
    if not col:
        col = list(range(A.shape[1]))

    nrows = A.shape[0]
    I = np.arange(nrows)[:, np.newaxis]
    A = np.concatenate((A, I), axis=1)
    A = np.asarray(sorted(A, key=operator.itemgetter(*col)))
    I = list(A[:, -1]) # get the index
    # convert to numeric if index in string
    for n, i in enumerate(I):
        if not isnumeric(i):
            I[n] = str2num(i)
    # I = I[:, np.newaxis]
    I = np.asarray(I)
    A = A[:, :-1]

    return A, I

def uniquerows(data, prec=5, sort=True):
    # d_r = np.fix(data * 10 ** prec) / 10 ** prec + 0.0
    if isinstance(data, (list, tuple)) or (isinstance(data, np.ndarray) and isrow(data)):
        data = np.asarray(data)[:, np.newaxis] # convert to a column vector
    b = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)
    c = np.unique(b).view(data.dtype).reshape(-1, data.shape[1])
    if not sort:
        ia, sorted_ia_index = sortrows(ia)
        c = c[sorted_ia_index,:]
        for n, k in enumerate(ic): # reindex
            ic[n] = int(np.where(sorted_ia_index == k)[0])
    return c, ia, ic

def poly2mask(r, c, m, n):
    """m, n: canvas size that contains this polygon mask"""
    fill_row_coords, fill_col_coords = polygon(r, c, (m, n))
    mask = np.zeros((m, n), dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def midpoint(v):
    """find the midpoints of a vector"""
    return v[:-1] + np.diff(v)/2.0


def SearchFiles(path, pattern, sortby='Name'):
    """
    Search for files
    sortby: 'Name', 'Modified Date', 'Created Date', 'Size'"""
    P = glob.glob(os.path.join(path, pattern))

    if isempty(P):
        return P, []

    N = [[]] * len(P)
    M = [[]] * len(P)
    C = [[]] * len(P)
    B = [[]] * len(P)
    for n, p in enumerate(P):
        N[n] = os.path.basename(os.path.normpath(p))
        M[n] = os.path.getmtime(p)
        C[n] = os.path.getctime(p)
        B[n] = os.path.getsize(p)

    # Sort
    if sortby == 'Name':
        pass
    elif sortby == 'Modified Date':
        P, N = zip(*[(x, y) for (z, x, y) in sorted(zip(M, P, N))])
    elif sortby == 'Created Date':
        P, N = zip(*[(x, y) for (z, x, y) in sorted(zip(C, P, N))])
    elif sortby == 'Size':
        P, N = zip(*[(x, y) for (z, x, y) in sorted(zip(M, P, N))])

    return P, N

def regexprep(STRING, EXPRESSION, REPLACE, N=None):
    """Function similar to MATLAB's regexprep, which allows replacement
    substitution of Nth occrrrence of character, which Python's re package
    lacks built-in. Note N's index start at 0 to be consistent with Python. An
    advantage of this design allows the user to specify negative index."""
    if N is None: # simply wrap the re.sub function
        return re.sub(EXPRESSION, REPLACE, STRING)
    else:
        indices = []
        for m in re.finditer(EXPRESSION, STRING):
            indices.append((m.start(), m.end(), m.group(0))) # start, end, whole match
        STRING = STRING[0:indices[N][0]] + REPLACE + STRING[(indices[N][1]+1):]
        return STRING


from bisect import bisect_left, insort
def medfilt1(x, N):
    """scipy.signal.medfilt is simply too slow on large kernel size or large
    data. This is an alternative. Not necessarily the same as MATLAB's
    implementation, but definitely faster than scipy's implementation.
    x: data
    N: order, or width of moving median
    """
    l = list(x[0].repeat(N))
    #l.sort()  # not needed because all values are the same here
    mididx = (N - 1) // 2
    result = np.empty_like(x)
    for idx, new_elem in enumerate(x):
        old_elem = x[max(0, idx - N)]
        del l[bisect_left(l, old_elem)]
        insort(l, new_elem)
        result[idx] = l[mididx]

    return result

def goodness_of_fit(xdata, ydata, popt, pcov, f0):
    """Calculate goodness of fit from curve_fit
    popt, pcov: returend by curve_fit
    f0: function used for fitting f(x, a, b, c, ...)
    p0: initial value used
    """
    yfit = f0(xdata, *popt)
    SSE = np.sum((yfit - ydata)**2)
    RMSE = np.sqrt(SSE/len(yfit))
    SS_total = np.sum((ydata - np.mean(ydata))**2)
    R_sq = 1.0 - SSE / SS_total
    R_sq_adj = 1.0 - (SSE/(len(xdata)-1)) / (SS_total/(len(xdata)-len(popt)-1))# Adjusted R_sq
    # R_sq_adj = 1-(1-R_sq)*(len(xdata-1))/(len(xdata)-len(popt)-1)
    gof = {'SSE': SSE, 'RMSE': RMSE, 'SS_total':SS_total, 'rsquare': R_sq, 'adjrsquare': R_sq_adj}

    return gof

def compare_goodness_of_fit(popt1, pcov1, popt2, pcov2, num_data_points, param_name=None, index=None):
    """Perform a t-test on a pair of fitted curves"""
    nvars = len(popt1)
    pcov1, pcov2 = np.sqrt(np.diag(pcov1)), np.sqrt(np.diag(pcov2))
    if index is not None:
        popt1, popt2 = popt1[index], popt2[index]
        pcov1, pcov2 = pcov1[index], popt2[index]


    T, df, P = [[]]*nvars, [[]]*nvars, [[]]*nvars
    for n, t1, v1, t2, v2 in enumerate(zip(popt1, pcov1, popt2, pcov2)):
        T[n]= (t1-t2) / np.sqrt(v1^2 + v2^2)
        df[n] = (num_data_points-nvars)*2
        P[n] = sp.stats.t.cdf(T, df=df)

    nvars = len(popt1) # update  for later looping

    if param_name is None:
        param_name = ['param_{:d}'.format(d) for d in range(nvars)]

    for n in range(nvars):
        print("{}: T = {:.4f}, df = {:d}, p = {:.4f}\n".format(param_name[n], T[n], df[n], P[n]))

    return T, df, P

def confidence_interval(ydata, popt, pcov, alpha=0.05, parameter_names=None):
    dof = max(0, len(ydata) - len(popt))
    tval = stats.distributions.t.ppf(1.0-alpha/2., dof)
    ci_list= []
    if parameter_names is None:
        for n, (p, var) in enumerate(zip(popt, np.diag(pcov))):
            sigma = var**0.5
            ci_list.append({'name': 'p{:d}'.format(n), 'mean': p, 'lower': p - tval * sigma, 'upper': p + tval * sigma})
    else:
        for pname, p, var in zip(parameter_names, popt, np.diag(pcov)):
            sigma = var**0.5
            ci_list.append({'name': pname, 'mean': p, 'lower': p - tval * sigma, 'upper': p + tval * sigma})

    return ci_list


def calculate_aic(n, mse, num_params):
    """calculate aic for regression
    * n: number of training examples
    * mse: mean squared error
    * num_params: number of parameters of the model
    """
	aic = n * np.log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
    """calculate bic for regression
    * n: number of training examples
    * mse: mean squared error
    * num_params: number of parameters of the model
    """
	bic = n * np.log(mse) + num_params * log(n)
	return bic

def serr(X, axis=0, toarray=False, printerr=False, returnOnError=None, *args, **kwargs):
    try:
        if toarray:
            X = np.array(X)
        return np.std(X, axis=axis, *args, **kwargs) / np.sqrt(np.shape(X)[axis])
    except Exception as err:
        if printerr:
            print(err)
        if returnOnError is None:
            return None
        elif returnOnError == 'first':
            return np.array(X)[0]

def diffx(X, axis=0, printerr=False, *args, **kwargs):
    """Wrapper function to deal with Panda's recent weird behavior"""
    def df_diff_apply(df):
        df_new = df.iloc[0, :]
        for c in df_new.index:
            if df[c].dtype.__str__() == 'object':
                pass
            else:
                df_new[c] = np.diff(df[c], axis=axis, *args, **kwargs)
                if len(df_new[c])==1:
                    df_new[c] = np.asscalar(df_new[c])

        return df_new
    try:
        return df_diff_apply(X)
    except Exception as err:
        if True:
            print(err)
        print(X)
        return None

def detect_outliers(x, percentile=[75, 25], factor=1.58, return_index=False, return_threshold=False):
    """
    Using R's boxplot.stats way of outlier detection for robustness
    Usages:
        * outliers = detect_outliers(x)
        * outliers, min_index, max_index = detect_outliers(x, return_index=True)
        * outliers, min_index, max_index = detect_outliers(x, return_threshold=True)
        * outliers, min_index, max_index, min_C, max_C = detect_outliers(x, return_index=True, return_threshold=True)
    """
    x = x[~np.isnan(x)]
    qq = np.percentile(x, percentile)
    q75, q25 = max(qq), min(qq)
    iqr = q75 - q25
    C = iqr * factor / np.sqrt(len(x))
    min_C = q25-C # Threshold for lower bound
    max_C = q75+C # Threshold for upper bound
    min_index = np.where(x<min_C) # Index of the lower bound outliers
    max_index = np.where(x>max_C) # Index of the upper bound outliers
    return_list = list(np.concatenate((np.array(x[min_index]),np.array(x[max_index]))))
    if return_index or return_threshold:
        return_list = [return_list]
    if return_index:
        return_list = return_list + [min_index] + [max_index]
    if return_threshold:
        return_list = return_list + [min_C] + [max_C]
    return  tuple(return_list)

def frequency_modulated_sine(f0, f, duration, ts, phase=0):
    """Return the frequency modulated sinosoidal wave
    f0: starting frequency [Hz]
    f: ending frequency [Hz]
    duration: duration of the wave [Sec]
    ts: sampling rate of wave [sec]
    phase: phase at the start of the wave, between [0, pi]
    """

    nu = np.linspace(f0, f, int(duration / ts) + 1)
    t = np.arange(0, duration+ts, ts)
    Y = np.sin(2 * np.pi * nu * t + phase)
    return t, Y

def softmax(X):
  exps = np.exp(X)
  return exps / np.sum(exps)

def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 100, fill = '█', mode='percentage'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        mode        - Optional  : display mode, either "percentage" or "counts"

    # Sample Usage:
        from time import sleep
        # A List of Items
        items = list(range(0, 57))
        l = len(items)

        # Initial call to print 0% progress
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i, item in enumerate(items):
            # Do stuff...
            sleep(0.1)
            # Update Progress Bar
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Sample Output
    Progress: |█████████████████████████████████████████████-----| 90.0% Complete
    Progress: |█████████████████████████████████████████████-----| 90/100 Complete
    """
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if mode == "percentage":
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    elif mode == "counts":
        print('\r%s  |%s| %d / %d %s' % (prefix, bar, iteration, total, suffix), end='\r')
    # Print New Line on Complete
    if iteration >= total:
        print('\nended')

def alpha(duration=400, amplitude=150, tau1=50, tau2=100, ts=0.1, force="positive"):
    """Returns a double exponential alpha function given parameters"""
    T = np.arange(0, duration+ts, ts)
    A = np.exp(-T/tau2) - np.exp(-T/tau1)
    if tau2==tau1:
        return np.zeros_like(T)
    elif force == "positive" and tau2<=tau1:
        # this will get us an alpha function with negative amp
        return np.zeros_like(T)
    elif force=="negative" and tau2>=tau1:
        # this will get us an alpha function with positive amp
        return np.zeros_like(T)

    A = A / np.max(np.abs(A)) * amplitude
    return A

def alphag(duration=400, amplitude=150, tau=100, ts=0.1):
    """Returns a single exponential alpha function given parameters"""
    T = np.arange(0, duration+ts, ts)
    G = (T/tau) * np.exp(1 - T/tau)
    G = G / np.max(np.abs(G)) * amplitude
    return G

def get_alpha_duration(A, ts=0.1, thresh=0.95):
    """Find duration of alpha function curve, when the value falls below a
    threshold fraction (e.g. 0.95 or about 3 taus) of the amplitude"""
    amplitude = np.max(A)
    amplitude_ind = np.argmax(A)
    index = np.where(A <= ((1.-thresh) * amplitude))[0]
    index = index[index > amplitude_ind]
    if not isempty(index):
        return index[0]*ts
    else:
        return None # beyond the scope of the curve. Cannot calculate


def fit_exp_with_offset(x, y, sort=False):
    """
    Fitting y = a * exp( b * x) + c
    using non-iterative method based on
    Regressions et Equations Integrales by Jean Jacquelin
    https://math.stackexchange.com/questions/2318418/initial-guess-for-fitting-exponential-with-offset
    """
    if sort:
        # Sorting (x, y) such that x is increasing
        X, _ = sortrows(np.c_[x, y], col=0)
        x, y = X[:, 0], X[:, 1]
    # Start algorithm
    S = np.zeros_like(x)
    S[1:] = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    S = np.cumsum(S)
    #for k in range(1, len(S)):
    #    S[k] = S[k-1] + 1/2 * (y[k] + y[k-1]) * (x[k] - x[k-1])

    M = np.empty((2, 2))
    N = np.empty((2, 1))

    # Getting b
    M[0, 0] = np.sum((x - x[0])**2)
    M[0, 1] = np.sum((x - x[0]) * S)
    M[1, 0] = M[0, 1]
    M[1, 1] = np.sum(S**2)


    N[0, 0] = np.sum((y - y[0]) * (x - x[0]))
    N[1, 0] = np.sum((y - y[0]) * S)

    B = np.matmul(np.linalg.inv(M),  N)
    b = B[1, 0]

    # Getting a and c
    M[0, 0] = len(x)
    M[0, 1] = np.sum(np.exp(b*x))
    M[1, 0] = M[0, 1]
    M[1, 1] = np.sum(np.exp(2*b*x))

    N[0, 0] = np.sum(y)
    N[1, 0] = np.sum(y * np.exp(b*x))
    AC = np.matmul(np.linalg.inv(M), N)
    c = AC[0, 0]
    a = AC[1, 0]

    return a, b, c

def fit_double_exp(x, y, sort=False):
    """
    Fitting y = b * exp(p * x) + c * exp(q * x)
    Implemented based on:
        Regressions et Equations Integrales by Jean Jacquelin
    """
    if sort:
        # Sorting (x, y) such that x is increasing
        X, _ = sortrows(np.c_[x, y], col=0)
        x, y = X[:, 0], X[:, 1]
    # Start algorithm
    n = len(x)
    S = np.zeros_like(x)
    S[1:] = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    S = np.cumsum(S)
    SS = np.zeros_like(x)
    SS[1:] = 0.5 * (S[:-1] + S[1:]) * np.diff(x)
    SS = np.cumsum(SS)

    # Getting the parameters
    M = np.empty((4, 4))
    N = np.empty((4, 1))

    M[:, 0] = np.array([np.sum(SS**2), np.sum(SS * S), np.sum(SS * x), np.sum(SS)])

    M[0, 1] = M[1, 0]
    M[1:,1] = np.array([np.sum(S**2),  np.sum(S * x), np.sum(S)])

    M[:2,2] = M[2, :2]
    M[2, 2] = np.sum(x**2)

    M[:3,3] = M[3,:3]
    M[3, 3] = n

    N[:, 0] = np.array([np.sum(SS * y), np.sum(S * y), np.sum(x * y), np.sum(y)])

    # Regression for p and q
    ABCD = np.matmul(np.linalg.inv(M), N)
    #set_trace()
    A, B, C, D = ABCD.flatten()
    p = 0.5 * (B + np.sqrt(B**2 + 4 * A))
    q = 0.5 * (B - np.sqrt(B**2 + 4 * A))

     # Regression for b, c
    I = np.empty((2, 2))
    J = np.empty((2, 1))

    beta = np.exp(p * x)
    eta  = np.exp(q * x)
    I[0, 0] = np.sum(beta**2)
    I[1, 0] = np.sum(beta * eta)
    I[0, 1] = I[1, 0]
    I[1, 1] = np.sum(eta**2)


    J[:, 0] = [np.sum(y * beta), np.sum(y * eta)]

    bc = np.matmul(np.linalg.inv(I), J)
    b, c = bc.flatten()

    return b, c, p, q


def fit_double_exp_with_offset(x, y, sort=False):
    """
    Fitting y = a + b * exp(p * x) + c * exp(q * x)
    Implemented based on:
        https://math.stackexchange.com/questions/2249200/exponential-regression-with-two-terms-and-constraints
    """
    if sort:
         # Sorting (x, y) such that x is increasing
        X, _ = sortrows(np.c_[x, y], col=0)
        x, y = X[:, 0], X[:, 1]
    # Start algorithm
    n = len(x)
    S = np.zeros_like(x)
    S[1:] = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    S = np.cumsum(S)
    SS = np.zeros_like(x)
    SS[1:] = 0.5 * (S[:-1] + S[1:]) * np.diff(x)
    SS = np.cumsum(SS)

    # Getting the parameters
    M = np.empty((5, 5))
    N = np.empty((5, 1))

    M[:, 0] = np.array([np.sum(SS**2), np.sum(SS * S), np.sum(SS * x**2), np.sum(SS * x), np.sum(SS)])

    M[0, 1] = M[1, 0]
    M[1:,1] = np.array([np.sum(S**2), np.sum(S * x**2), np.sum(S * x), np.sum(S)])

    M[0, 2] = M[2, 0]
    M[1, 2] = M[2, 1]
    M[2:,2] = np.array([np.sum(x**4),  np.sum(x**3), np.sum(x**2)])

    M[:3,3] = M[3,:3]
    M[3, 3] = M[4, 2]
    M[4, 3] = np.sum(x)

    M[:4, 4] = M[4, :4]
    M[4, 4] = n

    N[:, 0] = np.array([np.sum(SS * y), np.sum(S * y), np.sum(x**2 * y), np.sum(x * y), np.sum(y)])

    # Regression for p and q
    ABCDE = np.matmul(np.linalg.inv(M), N)
    A, B, C, D, E = ABCDE.flatten()
    p = 0.5 * (B + np.sqrt(B**2 + 4 * A))
    q = 0.5 * (B - np.sqrt(B**2 + 4 * A))

    # Regression for a, b, c
    I = np.empty((3, 3))
    J = np.empty((3, 1))

    I[0, 0] = n
    I[1, 0] = np.sum(np.exp(p * x))
    I[2, 0] = np.sum(np.exp(q * x))
    I[0, 1] = I[1, 0]
    I[1, 1] = np.sum(I[1, 0]**2)
    I[2, 1] = np.sum(I[1, 0] * I[2, 0])
    I[0, 2] = I[2, 0]
    I[1, 2] = I[2, 1]
    I[2, 2] = np.sum(I[2, 0]**2)

    J[:, 0] = [np.sum(y), np.sum(y * I[1, 0]), np.sum(y * I[2, 0])]

    abc = np.matmul(np.linalg.inv(I), J)
    a, b, c = abc.flatten()

    return a, b, c, p, q


def fit_gaussian_non_iter(x, y, sort=False):
    """
        Fitting Gaussian y = 1 / (sigma * sqrt(2 * pi)) * exp( -1/2 * ( (x - mu) / sigma )^2 )
        using non-iterative method based on
        Regressions et Equations Integrales by Jean Jacquelin
        """
    if sort:
        # Sorting (x, y) such that x is increasing
        X, _ = sortrows(np.c_[x, y], col=0)
        x, y = X[:, 0], X[:, 1]
    # Start algorithm
    S = np.zeros_like(x)
    S[1:] = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    S = np.cumsum(S)
    T = np.zeros_like(x)
    x_y = x * y
    T[1:] = 0.5 * ( x_y[:-1] + x_y[1:] ) * np.diff(x)
    T = np.cumsum(T)

    # S1 = np.zeros_like(x)
    # T1 = np.zeros_like(x)
    # for k in range(1, len(S1)):
    #    S1[k] = S1[k-1] + 1/2 * (y[k] + y[k-1]) * (x[k] - x[k-1])
    #    T1[k] = T1[k-1] + 1/2 * (y[k]*x[k] + y[k-1]*x[k-1]) * (x[k] - x[k-1])

    M = np.empty((2, 2))
    N = np.empty((2, 1))

    # Getting the parameters
    M[0, 0] = np.sum(S**2)
    M[0, 1] = np.sum(S * T)
    M[1, 0] = M[0, 1]
    M[1, 1] = np.sum(T**2)

    N[0, 0] = np.sum((y - y[0]) * S)
    N[1, 0] = np.sum((y - y[0]) * T)
    AB = np.matmul(np.linalg.inv(M), N)
    A = AB[0, 0]
    B = AB[1, 0]

    mu = - A / B

    sigma = np.sqrt(-1 / B)

    return mu, sigma

def pairwise_diff(A, B):
    """ Calculate the pair wise difference between each element of the
    two vectors of length N and M, respectively, and return a matrix of shape
    M x N
        * A: vector of length M
        * B: vector of length N
    """
    A = np.asarray(A).flatten()[:, np.newaxis]
    B = np.asarray(B).flatten()[np.newaxis,:]
    return A-B

def gaussian_kernel(ts, sigma=300., n=5, normalize=False):
    """Make gaussian kernel centered at 0
    ts: sampling rate [ms]
    n: use n standard deviations below and above 0 (mean).
    sigma: standard deviation (width of Gaussian kernel) [ms].
           During Up state, sigma = 10ms according to:
           Neske, G.T., Patrick, S.L., Connor, B.W. Contributions of
           Diverse Excitatory and Inhibitory Neurons to Recurrent Network
           Activity in Cerebral Cortex. The Journal of Nueroscience.
           35(3): 1089-1105 (2015). But this sd size may be too small for
           other processes. So default is set to 300ms for a smoother
           firing rate curve.
    normalize: Default False
        - 'area': normalize the area under the curve to be 1
        - 'peak': normalize the peak of curve to be 1
    """
    t = np.arange(-n*sigma, n*sigma+ts, ts)
    w = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-t**2/(2.*sigma**2))
    if normalize:
        if normalize == 'peak':
            w = w / np.max(w)
        elif normalize == 'area':
            w = w / np.sum(w)
        else:
            pass
    return(t, w)

def exists(obj):
    """Check if a variable exists"""
    if obj in locals() or obj in globals():
        return True
    else:
        return False

def spm_matrix_2d(P, order='T*R*Z*S'):
    """
    returns an affine transformation matrix

    Arguments:
        * P[0]  - x translation
        * P[1]  - y translation
        * P[2]  - rotation (radians)
        * P[3]  - x scaling
        * P[4]  - y scaling
        * P[5]  - affine
        * order (optional): application order of transformations.

    Returns:
        * A: affine transformation matrix

    To transform X = [x;y] coordinate, pad 1 at the end,
    i.e. X = [x; y; 1], then Y = A * X, where X, Y are 3 x n matrices
    """
    # Pad matrix in case only a subset is being specified
    q  = np.array([0, 0, 0, 1, 1, 0])
    P  = np.concatenate((P, q[(len(P)):]))
    T   =  np.array([
            [1,    0,      P[0]],
            [0,    1,      P[1]],
            [0,    0,      1]
            ])

    R   =  np.array([
            [np.cos(P[2]),   np.sin(P[2]),  0],
            [-np.sin(P[2]),  np.cos(P[2]),  0],
            [0,              0,             1]
            ])

    Z   =  np.array([
            [P[3],    0,    0],
            [0,      P[4],  0],
            [0,       0,    1]
            ])

    S   =  np.array([
            [1,    P[5],    0],
            [0 ,   1,       0],
            [0,    0,       1]
            ])

    order = order.replace('*', '@')
    A = eval(order)
     # Sanity check
    if not isnumeric(A).all() or A.ndim != 2 or any(np.array(A.shape) != 3):
        raise(ValueError('Order expression "{}" did not return a valid 3x3 matrix.'.format(order)))

    return A

def spm_matrix(P, order='T*R*Z*S'):
    """
    returns an affine transformation matrix

    Arguments:
        * P[0]  - x translation
        * P[1]  - y translation
        * P[2]  - z translation
        * P[3]  - x rotation about - {pitch} (radians)
        * P[4]  - y rotation about - {roll}  (radians)
        * P[5]  - z rotation about - {yaw}   (radians)
        * P[6]  - x scaling
        * P[7]  - y scaling
        * P[8]  - z scaling
        * P[9]  - x affine
        * P[10] - y affine
        * P[11] - z affine
        * order (optional) application order of transformations.

    Returns:
        * A: affine transformation matrix

    To transform X = [x;y;z] coordinate, pad 1 at the end,
    i.e. X = [x; y; z; 1], then Y = A * X, where X, Y are 4xn matrices
    """
    # Pad matrix in case only a subset is being specified
    q  = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    P  = np.concatenate((P, q[(len(P)):]))
    # Specify the matrices
    T   =  np.array([
            [1,    0,    0,    P[0]],
            [0,    1,    0,    P[1]],
            [0,    0,    1,    P[2]],
            [0,    0,    0,    1   ]
            ])
    R1  =  np.array([
            [1,    0,             0,             0],
            [0,    np.cos(P[3]),  np.sin(P[3]),  0],
            [0,   -np.sin(P[3]),  np.cos(P[3]),  0],
            [0,    0,             0,             1]
            ])

    R2  =  np.array([
            [np.cos(P[4]),  0,    np.sin(P[4]),  0],
            [0,             1,    0,             0],
            [-np.sin(P[4]), 0,    np.cos(P[4]),  0],
            [0,             0,    0,             1]
            ])

    R3  =  np.array([
            [np.cos(P[5]),  np.sin(P[5]),   0,   0],
            [-np.sin(P[5]), np.cos(P[5]),   0,   0],
            [0,             0,              1,   0],
            [0,             0,              0,   1]
            ])
    R = R1 @ R2 @ R3

    Z   =  np.array([
            [P[6],    0,    0,    0],
            [0,      P[7],  0,    0],
            [0,       0,    P[8], 0],
            [0,       0,    0,    1]
            ])

    S   =  np.array([
            [1,    P[9],   P[10],  0],
            [0,    1,      P[11],  0],
            [0,    0,      1,      0],
            [0,    0,      0,      1]
            ])

    order = order.replace('*', '@')
    A = eval(order)
    # Sanity check
    if not isnumeric(A).all() or A.ndim != 2 or any(np.array(A.shape) != 4):
        raise(ValueError('Order expression "{}" did not return a valid 4x4 matrix.'.format(order)))

    return A


def ks_test_survival(s1, s2, n, m, alpha=0.05, alpha_type=1):
    """
    Kolmogorov-Smirnov test on 2 survival curves
    """
    D = max(abs(s1 - s2))
    en = np.sqrt((n+m)/(n*m))
    c_alpha = np.sqrt(-1/2*np.log(alpha))
    D_critical = c_alpha * en
    if alpha_type==1:
        p = np.exp((D / en)**2 * (-2))
    else:
        p = stats.distributions.kstwobign.sf((1/en + 0.12+0.11 * en) * D)

    return D, p, D_critical

def faster_corr(X, Y):
    """
    faster way to compute Pearson's correlation
    between corresponding columns of two matrices
    http://stackoverflow.com/questions/9262933/what-is-a-fast-way-to-compute-column-by-column-correlation-in-matlab
    """
    X = X - np.mean(X, axis=0, keepdims=True) # zero mean
    Y = Y - np.mean(Y, axis=0, keepdims=True) # zero mean
    X = X / np.sqrt(np.sum(X**2, axis=0, keepdims=True)) # L2 normalization
    Y = Y / np.sqrt(np.sum(Y**2, axis=0, keepdims=True)) # L2 normalization
    R = np.sum(X*Y, axis=0)

    return R

def faster_cross_corr(X, Y):
    """
    faster way to compute Pearson's correlation
    between each combination of paris of two matrices
    returns a cross correlation matrix
    [[x1:y1, x1:y2, x1:y3,...., x1:yn],
     [x2:y1, x2:y2, x2:y3,...., x1:yn],
     [...]
     [xn:y1, xn:y2, xn:y3,...., xn:yn]
    """
    X = X - np.mean(X, axis=0, keepdims=True) # zero mean
    Y = Y - np.mean(Y, axis=0, keepdims=True) # zero mean
    X = X / np.sqrt(np.sum(X**2, axis=0, keepdims=True)) # L2 normalization
    Y = Y / np.sqrt(np.sum(Y**2, axis=0, keepdims=True)) # L2 normalization
    R = X.T.dot(Y)

    return R


def r2z(R, return_q=False):
    """
    Convert Pearson correlation to Z score and get a P value

    z = arctanh(R)
    p = Gaussian(mu=0, sigma=1)(z)
    q = NormCDF(mu=0, sigma=1)(z)
      = (1. + erf(z / sqrt(2.))) / 2.
      where erf = 2/sqrt(pi)*integral(exp(-t**2), t=0..z).

    Parameters
    ----------
    R : Pearson correlation between -1 and 1
    return_q: if return cumulative probability as well

    Returns
    -------
    z: z score
    p: p value
    q: cumulative distribution of the z score (only returned if return_q=True)
    """
    assert R <=1 and R>=-1

    z = np.arctanh(R)
    p = (1/np.sqrt(2*np.pi)) * np.exp(-z**2/2) # Gaussian N(0, 1)
    if return_q:
        q = stats.norm.cdf(z)
        return z, p, q
    else:
        return z, p

def sparse_memory_usage(mat):
    """
    Based on https://stackoverflow.com/questions/43681279/why-is-scipy-sparse-matrix-memory-usage-indifferent-of-the-number-of-elements-in
    """
    if mat.format in ('csr', 'csc'):
        return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
    elif mat.format == 'coo':
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
    elif mat.format == 'lil':
        return mat.data.nbytes + mat.rows.nbytes
    else:
        return sys.getsizeof(mat)

def cosine_similarities(mat):
    """
    Compute pairiwise cosine similarities between columns
    given sparse matrix mat
    """
    import sklearn.preprocessing as pp
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat

def cosine_similarities_ab(A, B):
    """
    Compute cosine similarities between matrix A and B.

    Parameters
    ----------
    A : np.ndarray
        Left matrix of shape (M x K).
    B : np.ndarray
        Right matrix of shape (N x K).

    Returns
    -------
    Cosine similarity matrix of shape (M x N).

    """
    # L2 normalize
    A = A / np.sqrt(np.sum(A**2, axis=1, keepdims=True))
    B = B / np.sqrt(np.sum(B**2, axis=1, keepdims=True))
    return A @ B.T

def jaccard_similarities(mat):
    """
    Compute pairwise Jaccard similarities between columns
    given sparse matrix mat
    """
    cols_sum = mat.getnnz(axis=0)
    ab = mat.T * mat

    # for rows
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    # for columns
    bb = cols_sum[ab.indices]

    similarities = ab.copy()
    similarities.data /= (aa + bb - ab.data)

    return similarities


def kl_divergence_pairwise(P):
    """
    Compute pairwise KL divergence in P, where
    each row of P is a probability distribution.
    
    Using formula:
    D_KL(P, Q) = H(P, Q) - H(P) 
               = (-\sum_i p_i \log q_i) - (-\sum p_i \log p_i)
    
    Returns an N x N matrix where N is the number of rows
    """
    H_p = -np.sum(P * np.log(P), axis=1, keepdims=True)
    H_p_q = -(P @ np.log(P.T))
    D_kl = H_p_q - H_p
    return D_kl


def harmonic_approx(n):
    """Returns an approximate value of n-th harmonic number.

       http://en.wikipedia.org/wiki/Harmonic_number
    """
    if n == 0:
        return 0
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def cubic_trig(b, c, d):
    """
    Trigonometric solution to a cubic function, assuming the form
    x^3 + b x^2 + c x + d = 0
    
    The coefficients can be of any shape (n1, n2, n3, ...), 
    and the solutions will be an output of dimension (n1, n2, n3, ..., 3)
    where the 3 solutions (both complex and real) are stacked in the 
    last dimension
    
    Reference: "A Generalized Trigonometric Solution of the Cubic Equation"
    https://www.jstor.org/stable/2968419?seq=1#metadata_info_tab_contents
    """
    # Transform to depressed form
    b_sq = b * b
    b_cb = b_sq * b
    b_3 = b/3
    p = c-b_sq/3
    q = 2/27*b_cb-1/3*b*c+d
    
    # Solve with complex trigonometric functions
    phi = 1/3*np.arcsin((3*q/(-2*p) * np.lib.scimath.sqrt(3/(-p))).astype(np.complex))
    f = -2 * np.sign(p) * np.lib.scimath.sqrt(-p/3)
    x1 = f * np.sin(phi) - b_3
    x2 = f * np.sin(phi + 2*np.pi/3) - b_3
    x3 = f * np.sin(phi + 4*np.pi/3) - b_3

    X = np.dstack([x1, x2, x3])
    return X

if __name__ == '__main__':
#    A = np.array([[2, 3], [1,2], [1, 2], [3, 2], [4,5], [3,1], [1,2], [2,3]])
#   A = ['a','b','a','c','a','b','c']
#    C, IA, IC = uniquerows(A)
    x = np.arange(0, 10, 0.1)
    #f0 = lambda x, a, b, c: a * np.exp(b*x)+c
    f0 = lambda x, a, b, c, p, q: a + b * np.exp(p * x) + c * np.exp(q * x)
    y = f0(x, 0, 2, 3, -1.6, 2) + np.random.randn(len(x))/10
    #a, b, c = fit_exp_with_offset(x, y)
    b, c, p, q = fit_double_exp(x, y)
    plt.scatter(x, y)
    plt.plot(x, f0(x, 0, b, c, p, q), 'r')
    plt.title('a={:.3f}, b={:.3f}, c={:.3f}'.format(0, b, c))

#    x = np.arange(-2.5, 7.1, 0.1)
#    f0 = lambda x, mu, sigma: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( -1/2 * ( (x - mu) / sigma )**2 )
#    y = f0(x, 2.08, 3.2) + np.random.randn(len(x))/100
#    mu, sigma= fit_gaussian_non_iter(x, y, sort=False)
#    plt.scatter(x, y)
#    plt.plot(x, f0(x, mu, sigma))
#    plt.title(r'$\mu$={:.3f}, $\sigma$={:.3f}'.format(mu, sigma))
