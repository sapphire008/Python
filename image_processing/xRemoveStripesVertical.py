import numpy as np
import pywt

def xRemoveStripesVertical(ima, decNum=8, wname='db42', sigma=8):
    """"
     Stripe and Ring artifact remover

     nima = xRemoveStripesVertical(ima, decNum, wname, sigma)

     Inputs:
       ima: image matrix
       decNum: highest decomposition level (L). Default 8.
       wname: wavelet type. See WFILTERS.
       sigma: damping factor of Gaussian function
           g(x_hat, y_hat) = 1 - exp(-y_hat^2 / (2 * sigma^2))
           Default 8.

     Output:
       nima: filtered image

     From
     Beat Munch, Pavel Trtik, Federica Marone, Marco Stampanoni. Stripe and
     ring artifact removal with combined wavelet -- Fourier filtering. Optics
     Express. 17(10): (2009)

     Suggestion for parameters:
     Based on the above cited paper,
       For waterfall artifacts (vertical stripes),
           decNum>=8, wname='db42', sigma>=8
       For ring artifacts
           decNum>=5, wname='db30', sigma>=2.4

    """

    # Check wavelet
    if wname not in pywt.wavelist():
        wname = pywt.Wavelet(wname, wfilters(wname))
    # wavelet decomposition
    Ch = [[]] * decNum # cell(1,decNum)
    Cv = [[]] * decNum # cell(1,decNum)
    Cd = [[]] * decNum # cell(1,decNum)
    for ii in np.arange(0, decNum):
        ima,Ch[ii],Cv[ii],Cd[ii] = pywt.dwt2(ima,wname)

    # FFT transform of horizontal frequency bands
    for ii in np.arange(0, decNum):
        # FFT
        fCv = np.fft.fftshift(np.fft.fft(Cv[ii]))
        my, mx = np.shape(fCv)

        # damping of vertical stripe information
        damp = 1-np.exp(-np.arange(-np.floor(my/2), -np.floor(my/2)+my-1, 1)**2/(2*sigma**2))
        fCv = fCv * damp[:,np.newaxis]

        # inverse FFT
        Cv[ii]=np.fft.ifft(np.fft.ifftshift(fCv))

    # wavelet reconstruction
    nima=ima
    for ii in np.arange(decNum, 0, -1):
        nima = nima[0:Ch[ii].shape[0], 0:Ch[ii].shape[1]]
        nima=pywt.idwt2((nima, (Ch[ii], Cv[ii], Cd[ii])), wname)
    return(nima)
