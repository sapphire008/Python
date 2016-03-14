"""PCA routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from kwiklib.utils.six.moves import range


# -----------------------------------------------------------------------------
# PCA functions
# -----------------------------------------------------------------------------
def compute_pcs(x, npcs=None, masks=None):
    """Compute the PCs of an array x, where each row is an observation.
    x can be a 2D or 3D array. In the latter case, the PCs are computed
    and concatenated iteratively along the last axis."""

    # Ensure x is a 3D array.
    if x.ndim == 2:
        x = x[..., None]
    assert x.ndim == 3
    # Ensure double precision
    x = x.astype(np.float64)

    nspikes, nsamples, nchannels = x.shape

    if masks is not None:
        assert isinstance(masks, np.ndarray)
        assert masks.ndim == 2
        assert masks.shape[0] == x.shape[0]  # number of spikes
        assert masks.shape[1] == x.shape[2]  # number of channels

    # Compute regularization cov matrix.
    if masks is not None:
        unmasked = masks > 0
        # The last dimension is now time. The second dimension is channel.
        x_swapped = np.swapaxes(x, 1, 2)
        # This is the list of all unmasked spikes on all channels.
        # shape: (n_unmasked_spikes, nsamples)
        unmasked_all = x_swapped[unmasked, :]
        # Let's compute the regularization cov matrix of this beast.
        # shape: (nsamples, nsamples)
        cov_reg = np.cov(unmasked_all, rowvar=0)
    else:
        cov_reg = np.eye(nsamples)
    assert cov_reg.shape == (nsamples, nsamples)

    pcs_list = []
    # Loop over channels
    for channel in range(nchannels):
        x_channel = x[:, :, channel]
        # Compute cov matrix for the channel
        if masks is not None:
            # Unmasked waveforms on that channel
            # shape: (n_unmasked, nsamples)
            x_channel = np.compress(masks[:, channel] > 0,
                                           x_channel, axis=0)
        assert x_channel.ndim == 2
        # Don't compute the cov matrix if there are no unmasked spikes
        # on that channel.
        alpha = 1. / nspikes
        if x_channel.shape[0] <= 1:
            cov = alpha * cov_reg
        else:
            cov_channel = np.cov(x_channel, rowvar=0)
            assert cov_channel.shape == (nsamples, nsamples)
            cov = alpha * cov_reg + cov_channel
        # Compute the eigenelements
        vals, vecs = np.linalg.eigh(cov)
        pcs = vecs.T.astype(np.float32)[np.argsort(vals)[::-1]]
        # Take the first npcs components.
        if npcs is not None:
            pcs_list.append(pcs[:npcs,...])
        else:
            pcs_list.append(pcs)
    # Return the concatenation of the PCs on all channels, along the 3d axis,
    # except if there is only one element in the 3d axis. In this case
    # we convert to a 2D array.
    pcs = np.dstack(pcs_list)
    assert pcs.ndim == 3
    if pcs.shape[2] == 1:
        pcs = pcs[:, :, 0]
        assert pcs.ndim == 2
    return pcs

def project_pcs(x, pcs):
    """Project data points onto principal components.

    Arguments:
      * x: a 2D array.
      * pcs: the PCs as returned by `compute_pcs`.

    """
    x_proj = np.einsum('ijk,jk->ki', pcs, x)  # Notice the transposition.
    x_proj *= 100.
    return x_proj


