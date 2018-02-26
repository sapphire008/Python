# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:41:55 2015

Python implemented MATLAB utilities

@author: Edward
"""

import numpy as np
import scipy as sp
from skimage.draw import polygon
import re
import glob
import os
import operator
import pandas as pd
from collections import OrderedDict
from pdb import set_trace

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

def str2num(lit):
    """MATLAB behavior of str2num.
    str2num('1') --> 1
    str2num('[5,3,2]') --> [5,3,2]
    str2num('[5,3,2;2,3,1]') --> [[5,3,2],[3,2,1]]
    """
    # Separate the string by semicolon ";"
    lit = lit.split(";")
    # Identify all numbers
    lit = [re.findall(r"[-+]?\d*\.\d+|\d+", l) for l in lit]
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
        return True if m else False

def isnumber(obj):
    """Determine if the object is a single number"""
    attrs = ['__add__', '__sub__', '__mul__', '__pow__', '__abs__']
    return all(hasattr(obj, attr) for attr in attrs)

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

def isvector(v):
    v = np.asarray(v)
    return True if len(v.shape) == 1 or v.shape[0] == 1 or v.shape[1] == 1 else False

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
    df = df.loc[~np.isnan(df['value'])]
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
    """sortby: 'Name', 'Modified Date', 'Created Date', 'Size'"""
    P = glob.glob(path+pattern)

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
    f0: function used for fitting
    p0: initial value used
    """
    yfit = f0(xdata, *popt)
    SSE = np.sum((yfit - ydata)**2)
    RMSE = np.sqrt(SSE/len(yfit))
    SS_total = np.poly1d(np.polyfit(xdata, ydata, 1))
    SS_total = np.sum((SS_total(xdata) - ydata)**2)
    R_sq = 1.0 - SSE / SS_total
    R_sq_adj = 1.0 - (SSE/(len(xdata)-len(popt))) / (SS_total/(len(xdata)-1))# Adjusted R_sq
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


def serr(X, axis=0, *args, **kwargs):
    return np.std(X, axis=axis, *args, **kwargs) / np.sqrt(np.shape(X)[axis]) 


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
    
   

if __name__ == '__main__':
#    A = np.array([[2, 3], [1,2], [1, 2], [3, 2], [4,5], [3,1], [1,2], [2,3]])
#   A = ['a','b','a','c','a','b','c']
#    C, IA, IC = uniquerows(A)
    t, Y = frequency_modulated_sine(f0=0, f=10, duration=10, ts=0.0001, phase=0)
    plt.plot(t, Y)
