# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:41:55 2015

Python implemented MATLAB utilities

@author: Edward
"""

import numpy as np
from skimage.draw import polygon
import re
import glob
import os
import operator
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
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
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

def isempty(m):
    """Return true if:
    a). an empty string
    b). a list of length zero
    c). a tuple of length zero
    d). a numpy array of length zero
    e). a singleton that is not None
    """
    if isinstance(m, (list, tuple, str, np.ndarray)):
        if len(m) == 0:
            return True
        else:
            return all([not x for x in m])
    else:
        return True if m else False

def isnumeric(obj):
    """Check if an object is numeric, or that elements in a list of objects 
    are numeric"""
    def f(x):
        attrs = ['__add__', '__sub__', '__mul__', '__pow__', '__abs__']
        return all(hasattr(x, attr) for attr in attrs)
    
    # Allow application to iterables
    f_vec = np.frompyfunc(f, 1, 1)
    tf = f_vec(obj)
    if isinstance(tf, np.ndarray):
        tf = tf.astype(dtype=bool)
    return tf
    
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


def sort_nicely( l ):
    """ Sort the given list of strings in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l     
    
    
def sortrows(A, col=None):
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
    

def unique(A, byrow=False, occurrence='first', stable=False):
    """MATLAB's unique. Can apply to both numerical matrices and list of 
    strings. Not for performance, but for better outputs
    Inputs:
        A: list, list of list, numpy nadarray
        byrow: [True,False]. If True, the matrix A will be converted into
                a list, column wise. If False, returns the unique rows of A.
        occurence: ['first', 'last']. Specify which index is returned in IA in 
                    the case of repeated values (or rows) in A. The default 
                    value is OCCURENCE = 'first', which returns the index of 
                    the first occurrence of each repeated value (or row) in A, 
                    while OCCURRENCE = 'last' returns the index of the last 
                    occurrence of each repeated value (or row) in A.
        stable: [True, False]. Whether or not sort the result C. If True, 
                returns the values of C in the same order that they appear in 
                A; if False, returns the values of C in sorted order. If A is 
                a row vector, then C will be a row vector as well, otherwise C
                will be a column vector. IA and IC are column vectors. If 
                there are repeated values in A, then IA returns the index of 
                the first occurrence of each repeated value.
    
    Return: 
        C: list of unique items
        IA: index of ('first', 'last', specified by occurrence parameter)
            occurence, such that C = A[IA]
        IC: index such that A = C[IC]
    """
    # convert to numpy array for easier manipulation
    A = np.asarray(A, order='F')
    
        # Return if there is only 1 item
    if A.size < 2:
        return A
    
    iscolvec = iscol(A) # take note the shape of input
    
    if byrow and not ismatrix(A):
        # call the function itself without sorting by rows
        return unique(A, byrow=False, occurrence=occurrence, stable=stable)
    
    if not byrow:
        # if not by row, convert to column vector
        A = A.flatten(order='F')[:, np.newaxis]
        nRows = A.size
    else:
        nRows, _ = A.shape
        
    # Sort the input
    sortA, indSortA = sortrows(A, col=None)

    # groupsSortA indicates the location of the non-matching entires
    groupsSortA = sortA[:-1, :] != sortA[1:, :]
    groupsSortA = groupsSortA.any(axis=1) # row vector
        
    if occurrence == 'last':
        groupsSortA = np.append(groupsSortA, True) # Final element is always a member of unique list.
    else: # occurrence == 'first' or stalbe==True
        groupsSortA = np.insert(groupsSortA, 0, True) # Final element is always a member of unique list.
        
    # Extract unique elements
    if stable:
        invIndSortA = indSortA
        invIndSortA[invIndSortA] = np.arange(nRows) # Find inverse permutation
        logIndA = groupsSortA[invIndSortA] # Create new logical by indexing into groupsSortA
        C = A[logIndA, :] 
        IA = np.where(logIndA)[0] # Find the indices of the unsorted logical
    else:
        C = sortA[groupsSortA, :]
        IA = indSortA[groupsSortA] # Find the indices of the sorted logical
    
    # Find IC
    IC = np.zeros(nRows, dtype=np.int64)
    for n, a in enumerate(A):
        IC[n] = int(np.where((C==a).all(axis=1))[0])
        
    # If A is column vector, return C as column vector
    if iscolvec:
        C = C[:, np.newaxis]
        
    
    return C, IA, IC, groupsSortA, sortA, indSortA
    

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
        
        
if __name__ == '__main__':
    A = np.array([[2, 3], [1,2], [1, 2], [3, 2], [4,5], [3,1], [1,2], [2,3]])
    C, IA, IC, groupsSortA, sortA, indSortA = unique(A, byrow=True, occurrence='last', stable=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
