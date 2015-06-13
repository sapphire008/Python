# -*- coding: utf-8 -*-
"""
Created on Mon May 05 22:57:16 2014

@author: Edward
"""

import numpy as np

# creating ndarray
data1 = [6, 7.5, 8, 0.1]
arr1 = np.array(data1)

arr1.shape# array size
arr1.dtype# data type: default float64

data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2.shape

arr3 = np.zeros(10) # create 1x10 array of zeros
arr4 = np.ones((5,5)) # create 5x5 array of ones
arr4 = np.empty((2,3,2)) # create 2x3x2 array of empty
# shape argument needs to be a tuple
# np.empty will return garbage values, insteaad of zeros
arr5 = np.arange(15) # create an array from 0 to 14, like range in Python
arr6 = np.ones_like(arr5) # take arr5 and build an array of ones the same shape and type as arr5
arr7 = np.zeros_like(arr5) #take arr5 and build an array of zeros the same shape and type as arr5
arr8 = np.empty_like(arr5) #take arr5 and build an array of empty the same shape and type as arr5

arr9 = np.eye(5)# create 5x5 identity matrix
arr10 = np.identity(5) #same as np.eye 

# set and cast data type
arr1 = np.array([1,2,3],dtype=np.float64)
arr2 = arr1.astype(np.int32) # cast to int32
arr3 = np.array(['1.323','-4.23323','5.907','-42.3243'],dtype = np.string_)
arr4 = arr3.astype(np.float32) # string to number

# array element by element operations
arr = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
arr * arr # elment by element multiplication
arr ** 0.5 # element by element raising power to 0.5 (taking square root)
1/arr # element by element inversion: note that the precision will be different
# if carried division on np.int32 or other integer types. The division will be
# essentially a floor division

# indexing and slicing
arr = np.arange(10)
arr[5]# get the 6th element, as index starts at 0
arr[5:8] # get sub-indices of elements
arr[5:9] = 70 # assign index 5 to 9 to 70

arr_slice = arr[5:8] # this creates an aliased view of the arr at slices/indices 5:8
arr_slice[1]=12345
arr # arr[5] will be changed to 12345
# to trutly copy the data
arr_new_slice = arr[5:8].copy()

# 2D array
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2] #will get all the elements in row (axis 0) index 2 (3rd row)
arr2d[2][1] # get row (axis 0) index 2 element (axis 1) index 1 = 8
arr2d[2,1] # same as above
arr2d[:2,1:] # along axis 0, get up to (right before) index 2, at axis 1, get from index 1 to the end

# boolean indexing
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = randn(7,4)
#suppose each row of data corresponds to names
# get all the values that are corresponding to 'Bob'
data[names=='Bob']
# get the values starting from the 3rd column
data[names=='Bob',2:]
# for 'Not' operation, use "-"
data[-(names=='Bob')]
data[names!='Bob'] # same as above
# use & for AND operation and | for OR operation

# Fancy indexing: using integer arrays. This will always copy the data
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i

arr[[4,3,0,6]] # index from the beginning
arr[[-3, -5, -7]] # index from the end

arr = np.arange(32).reshape((8,4))
arr[[1,5,7,2],[0,3,1,2]] # pairwise indexing: (1,0), (5,3), ... selected
arr[np.ix_([1,5,7,2],[0,3,1,2])] # selecting a square region

# transposing
arr = np.arange(15).reshape((3,5))
arr.T
# compute inner product of matrices X'*X
arr = np.random.randn(6,3)
np.dot(arr.T,arr)
# permute for higher dimensions: np.transpose
arr = np.arange(16).reshape((2,2,4))
arr.transpose((1,0,2))
# swap axis
arr.swapaxes(1,2)

# Example data processing using ndarrays
points = np.arange(-5,5,0.01) # 1000 eqully spaced points
xs, ys = np.meshgrid(points,points)
import matplotlib.pyplot as plt
z = np.sqrt(xs**2 + ys**2)
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')

# x if condition else y alternative
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
# suppose take value from xarr if cond is true, otherwise, take from yarr
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
#equivalently, alternatively, and more efficiently for large arrays
result = np.where(cond,xarr,yarr)
# another example: producing an array based on threshold
arr = randn(4,4)
np.where(arr>0,2,-2)# if arr is greater than 0, insert 2, otherwise, insert -2
# consider Excel's if function: if(boolean_testing, value_if_true, value_if_false)
# both np.where and Excel's if function can be nested, e.g.
cond1 = randn(1,100)>0
cond2 = randn(1,100)<0
result = np.where(cond1 & cond2,0, np.where(cond1,1,np.where(cond2,2,3)))
#alternatively, use boolean expression
result = 1*cond1 + 2*cond2 + 3* -(cond1 | cond2) #but more difficult to come to first thought

# statistical:
np.mean(arr,axis=0)
np.std(arr,ddof=1)#denominator: N - ddof, default 0

# boolean arrays
bools = np.arry([True,False,False,True,False])
bools.all
bools.any

# sorting
arr = randn(8)
arr.sort() # will return a copy of the sorted array
# unique
arr.unique() # applies to both numerics and strings

# save binary data
arr = np.arrange(10)
np.save('some_array.npy',arr) # save on the disk, if name not appended .npy, will append automatically
np.load('som_array.npy') # load array
np.savez('array_archive.npz',a=arr,b=arr) # save as a zip archive of arrays
#when loading an archive, the result will be a dictionary like object
#load text file
arr = np.loadtxt('array_ex.txt',delimiter=',')

# Linear Algebra
x = np.array([[1.,2.,3.],[4.,5.,6.]])
y = np.array([[6.,23.],[-1.,7.],[8.,9.]])
x.dot(y)
#equivalently
np.dot(x,y)

from numpy.linalg import inv, qr
X = rand(5,5)
mat = X.T.dot(X)
inv(mat) # inverse matrix
q,r = qr(mat) # QR factorization / orthogonaliation

# random number generation
samples = np.random.normal(size=(4,4))

# Example Random Walk
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2,size=(nwalks,nsteps))
steps = np.where(draws>0,1,-1)
walks = steps.cumsum(1)
# get max and min of all walks
walk_max, walk_min = walks.max(), walks.min()
# compute minimum crossing time to 30 or -30
hits30 = (np.abs(walks)>=30).any(1).sum()
crossing_times = (np.abs(walks)>=30).argmax(1).mean()

# This concludes today's study