def wfilters(wname):
    """Extended list of filter parameters, borrowed from MATLAB's wfilters
       [LO_D,HI_D,LO_R,HI_R] = WFILTERS('wname') computes four
       filters associated with the orthogonal or biorthogonal
       wavelet named in the string 'wname'.
       The four output filters are:
           LO_D, the decomposition low-pass filter
           HI_D, the decomposition high-pass filter
           LO_R, the reconstruction low-pass filter
           HI_R, the reconstruction high-pass filter
       Available wavelet names 'wname' are:
       Daubechies: 'db1' or 'haar', 'db2', ... ,'db45'
       Coiflets  : 'coif1', ... ,  'coif5'
       Symlets   : 'sym2' , ... ,  'sym8', ... ,'sym45'
       Discrete Meyer wavelet: 'dmey'
       Biorthogonal:
           'bior1.1', 'bior1.3' , 'bior1.5'
           'bior2.2', 'bior2.4' , 'bior2.6', 'bior2.8'
           'bior3.1', 'bior3.3' , 'bior3.5', 'bior3.7'
           'bior3.9', 'bior4.4' , 'bior5.5', 'bior6.8'.
       Reverse Biorthogonal:
           'rbio1.1', 'rbio1.3' , 'rbio1.5'
           'rbio2.2', 'rbio2.4' , 'rbio2.6', 'rbio2.8'
           'rbio3.1', 'rbio3.3' , 'rbio3.5', 'rbio3.7'
           'rbio3.9', 'rbio4.4' , 'rbio5.5', 'rbio6.8'.

       [F1,F2] = WFILTERS('wname','type') returns the following
       filters:
       LO_D and HI_D if 'type' = 'd' (Decomposition filters)
       LO_R and HI_R if 'type' = 'r' (Reconstruction filters)
       LO_D and LO_R if 'type' = 'l' (Low-pass filters)
       HI_D and HI_R if 'type' = 'h' (High-pass filters)

       M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
       Last Revision 08-May-2012.
       Copyright 1995-2012 The MathWorks, Inc.
    """
    return({
    'db1':,
    'haar':,
    'db2':,
    'db3':,
    'db4':,
    'db5':,
    'db6':,
    'db7':,
    'db8':,
    'db9':,
    'db10':,
    'db11':,
    'db12':,
    'db13':,
    'db14':,
    'db15':,
    'db16':,
    'db17':,
    'db18':,
    'db19':,
    'db20':,
    'db21':,
    'db22':,
    'db23':,
    'db24':,
    'db25':,
    'db26':,
    'db27':,
    'db28':,
    'db29':,
    'db30':,
    'db31':,
    'db32':,
    'db33':,
    'db34':,
    'db35':,
    'db36':,
    'db37':,
    'db38':,
    'db39':,
    'db40':,
    'db41':,
    'db42':,
    'db43':,
    'db44':,  
    'db45':

    })
