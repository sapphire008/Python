# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""
import numpy as np
from MATLAB import *
from ImportData import *
from matplotlib import pyplot as plt

from pdb import set_trace

def MPI(Img):
    """Create Maximum Projection Intensity image from a stack of image
    Assuming z dimension is the third dimension"""
    return np.squeeze(np.max(Img, axis=2))

def FTimeSeries(Img, mask, mode=['F', 'dF','dF/F', 'hist']):
    """Extract the time series from a image series given binary ROI mask"""
    if 0 < Img.ndim - mask.ndim < 2:
        mask = mask[:,:,np.newaxis]
    
    if isinstance(mode, str):
        mode = [mode]

    # Get data at ROI
    Img_masked = Img * mask
    Img_masked = np.ma.masked_where(Img_masked==0, Img_masked)
    # Dimensions
    for m in range(Img_masked.ndim-1):
        if m == 0:
            F = Img_masked
        
        F = F.mean(axis=0)
    
    # Calculate 
    F = np.asarray(F)
    Fout = dict()
    for m in mode:
        if m == 'F':
            Fout[m] = F
        elif m == 'dF': # from Ben
            Fout[m] = F - F[1]
        elif m == 'dF/F': # from Ben
            Fout[m] = (F - F[1]) / F[1] * 100
        elif m == 'hist':
            I = Img_masked.compressed()
            n, b = np.histogram(I, int(len(I)/10)) # count, bin
            b = midpoint(b)
            Fout[m] = (n, b) 
            
    return Fout
    
def makeSquareROIMask(roi, m, n):
    """
    roi: roi object
    m: number of rows of the canvas
    n: number of columns of the canvas
    """
    x = roi.position[:,0].squeeze()
    x = np.array([x[0], x[0], x[1], x[1]], x[0])
    y = roi.position[:,1].squeeze()
    y = np.array([y[0], y[1], y[1], y[0]], y[0])
    mask = poly2mask(y, x, m, n)
    return mask

if __name__ == '__main__':
    # load the image
    img = 'D:/Data/2photon/2016/05.May/Image 4 May 2016/Slice B/Slice B CCh Double.512x200y75F.m1.img'
    zImage = ImageData(dataFile=img, old=True)
    # Load ROI
    roifile = 'D:/Data/2photon/2016/05.May/Image 4 May 2016/Slice B/Slice B Triple Doublet.512x200y75F.m2 ROI.roi'
    ROI = ROIData(roifile, old=True)
    # Get time series based on ROI
    m, n = zImage.Protocol.Height, zImage.Protocol.Width
    RESULTS = []
    for k, r in enumerate(ROI):
        # Convert ROI to mask
        x = r.position[:,0].squeeze()
        x = np.array([x[0], x[0], x[1], x[1]], x[0])
        y = r.position[:,1].squeeze()
        y = np.array([y[0], y[1], y[1], y[0]], y[0])
        mask = poly2mask(y, x, m, n)
        Fout = FTimeSeries(zImage.img, mask, mode=['F','dF','dF/F','hist'])
        RESULTS.append( Fout )
        
    plt.imshow(zImage.img[:,:,0] * (1-mask))
                           
        
        