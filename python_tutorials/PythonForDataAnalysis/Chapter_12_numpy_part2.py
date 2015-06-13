# -*- coding: utf-8 -*-
"""
Created on Mon May 12 02:45:05 2014
Modified on Thur May 22 12:12:05 2014
@author: Edward
"""

import numpy as np
from numpy.random import randn

# Reshape
arr = np.arange(8)
arr.reshape((4,2))
arr.reshape((4,2)).reshape((2,4))
#can specify only one dimension, and let the program infer the other dimension
#by passing a -1
arr = np.arange(15)
arr.reshape((5,-1))

# falttening / raveling
arr = np.arange(15).reshape((5,3))
arr.ravel() # does not copy data
arr.flatten()

# row major vs. column  major: C vs. Fortran data order
# row major: data of each row are stored adjacently in the memory locations; 
# --> rule used by C, and default of numpy
# column major: data of each column are stored adjacently in the memory
# locations; --> rule used by Fortran, and default of MATLAB
arr = np.arange(12).reshape((3,4))
arr.ravel() # taking by rows, taking higher dimensions first, then to axis 0
#vs
arr.ravel('F') #taking by columns, taking lower dimensions first
# to set specific orders, set order when assinging the varibable
arr_c = np.ones((1000,1000), order = 'C')
arr_c.flags #check variable attribute
arr_f = np.ones((1000,1000), order = 'F')
arr_f.flags #check variable attribute
# to make a copy of the array with converted order
new_arr_c = arr_f.copy('C')
new_arr_c.flags
# In general, continguous data performs better than non-contiguous data.
# For instance, to sum the aray over the rows, C_contiguous data runs faster
# than F_contiguous data

# concatenation and splitting
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8,9],[10,11,12]])
np.concatenate([arr1,arr2],axis=0) # concatenate along first dimension / rows
# alternatively
np.vstack((arr1,arr2)) #vertically stack
np.column_stack((arr1,arr2))
np.concatenate([arr1,arr2],axis=1) # concatenate along second dimension /columns
# alternatively
np.hstack((arr1,arr2)) #horizontally stack
np.row_stack((arr1,arr2))
# splitting arrays
arr = randn(5,2)
first, second, third = np.split(arr, [1,3])
# stacking helpers: r_ and c_
arr1 = np.arange(6).reshape((3,2))
arr2 = randn(3,2)
np.r_[arr1,arr2] # row concatenation
np.c_[np.r_[arr1,arr2],np.arange(6)] #column concatenation
np.c_[1:6,-10:-5]#translate slices/MATLAB-like vector notation into arrays

# repeating notations: tile and repeat
arr = np.arange(3)
arr.repeat(3) # repeat each element 3 times, concatenate each repeats along the row (over columns)
#array([0, 0, 0, 1, 1, 1, 2, 2, 2]) --> different from MATLAB's repmat
#can specify how many times each element repeat
arr.repeat([2,3,4])# repeat the 1st element twice, 2nd three times, and 3rd four times
# can repeat along a particular axis
arr = randn(2,2)
arr.repeat(2,axis=0) #repeaet along rows
# tile is the same as MATLAB's repatm, which stacks copies of the array
np.tile(arr,2) #default along the row/axis=0
# much like repmat, can specify the layout of the repeats
np.tile(arr,(3,2))#repeat 3 times along the row, and twice along the column

# Fancy indexing equivalents: take and put
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds] # fancy indexing
arr.put(inds, [40,41,42,43])#put [40, 41, 42,43] in place of arr[inds], one by one
# this method does not return anything, but will modify the array arr
arr.take(inds,axis=0) # get the elements of the array at the indices.
# This is the same as arr[ind]. Performance for each is relatively the same,
# but take may be slightly better (10s~20s of nanoseconds)

# Broadcasting
arr = randn(4,3)
arr.mean(0)#take the mean along axis=0
dmeaned = arr - arr.mean(0) #broadcast along the rows, so that from each element,
# the mean of its respective column is removed
dmeaned = arr - arr.mean(1).reshape((4,1)) #broadcast along the columns, so that from each element,
# the mean of its respective row is removed. Note that we have to reshape
# the array into a volumn vector to allow broadcasting to occur. Tranpose will
# not work!
# adding new axis: np.newaxis
arr = np.zeros((4,4))
arr_3rd = arr[:, np.newaxis, :]
arr_1d = np.random.normal(size=3)
arr_1d[:,np.newaxis] # transpose row vector into column vector
# suppose we have a 3D array, and we want to demean axis=2 (3rd dimension)
arr = randn(3,4,5)
demeaned = arr - arr.mean(2)[:,:,np.newaxis]
# The following function generalize demean procedue without sacraficing performance
def demean_axis(arr, axis=0):
    # demean without sacraficing performance
    means = arr.mean(axis) # get the mean to be removed
    #This generalized things like [:,:,np.newaxis] to N dimensions
    indexer = [slice(None)] * arr.ndim #create slicer object with the same dimension as arr
    indexer[axis] = np.newaxis #set the axis at which demean needs to be performed to np.newaxis
    return arr - means[indexer]#demean

# ufunc

# reduce: aggregate values by successive applications of operation
# reduceat(x,bins): local reduce or group by.
arr = np.arange(10)
np.add.reduce(arr) # aggregates values, along an axis
# equivalent to 
np.sum(arr)
arr = randn(5,5)
arr[::2].sort(1) # sort row 0 to row 2
arr[:,:-1] < arr[:,1:] # for each row, compare if the last numbers are smaller the first numbers, i.e. sorted
np.logical_and.reduce(arr[:,:-1]<arr[:,1:],axis=1)

# accumulate: aggregate values, preserving all partial aggregates
arr = np.arange(15).reshape((3,5))
#accumulate: produce intermediate accumulated values, comparable to cumsum
np.add.accumulate(arr,axis=1)

# outer: apply operations to all pairs of elements in x and y. Result array has shape x.shape + y.shape
arr = np.arange(3).repeat([1,2,2])
np.multiply.outer(arr,np.arange(5))
result = np.subtract.outer(randn(3,4), randn(5))
# result of outer will have the dimension the sum of the inputs
# note that the sum of tuples (which wha shapes are) are simly concatenating
# the tuples
result.shape
#(3,4,5)

# custom ufuncs: useful but slower than numpy's C based library
def add_element(x,y):
    return x+y

add_them = np.frompyfunc(add_element, 2,1)
add_them(np.arange(8),np.arange(8))#returns python object

add_them = np.vectorize(add_element, otypes=[np.float64])
add_them(np.arange(8),np.arange(8)) #return array

# Sorting
arr = randn(3,5)
arr[:,0].sort() # sort first column values, original modified
np.sort(arr) # creates a new copy
np.sort(arr)[:,::-1]# sort in descending order

# argsort and lexsort
# returns indices of the array after sorting
values = np.array([5,0,1,3,2])
indexer = values.argsort()
#example: reorder 2D array by its first row
arr = randn(3,5)
arr[0] = values
arr[:,arr[0].argsort()]

# lexsort: performs lexicographical sort, sort in paris
# sort last array passed first, then go backwards
first_name = np.array(['Bob','Jane','Steve','Bill','Barbra'])
last_name = np.array(['Jone','Arnold','Arnold','Jones','Walters'])
sorter = np.lexsort((first_name, last_name))
zip(last_name[sorter],first_name[sorter])

# alternate sort algorithms
# stable sort
values = np.array(['2:first','2:second','1:first','1:second','1:third'])
key = np.array([2,2,1,1,1])
indexer = key.argsort(kind='mergesort')
values.take(indexer)
# find elements in sorted array
arr = np.array([0, 1, 7, 12, 15])
# searchsorted: perform binary search on sorted array, return index where
# the value passed onto the searchsorted method need to be in order to maintain
# the sort
arr.searchsorted(9)

# NumPy Matrix Class
Xm = np.matrix(randn(5,5))
Ym = Xm[:, 0]
Ym.T * Xm *Ym # multiplication
Xm.I * Xm # inverse multiplication

# Memory-mapped files: handling files that are too large to be loaded to the RAM
filename = 'mymap'
#create a memory mapped file called mymap
mmap = np.memmap(filename,dtype = 'float32',mode = 'w+', shape=(10000,10000))
#take a slice
section = mmap[:5]
#assign values to the slice
section[:] = np.random.randn(5, 10000)
mmap.flush # this will write the slice onto the disk

# This concludes today's study