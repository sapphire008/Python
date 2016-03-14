"""PCA tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from spikedetekt2.processing import compute_pcs, project_pcs


# -----------------------------------------------------------------------------
# PCA tests
# -----------------------------------------------------------------------------
def test_compute_pcs():
    """Test PCA on a 2D array."""
    # Horizontal ellipsoid.
    x = np.random.randn(20000, 2) * np.array([[10., 1.]])
    # Rotate the points by pi/4.
    a = 1./np.sqrt(2.)
    rot = np.array([[a, -a], [a, a]])
    x = np.dot(x, rot)
    # Compute the PCs.
    pcs = compute_pcs(x)
    assert pcs.ndim == 2
    assert (np.abs(pcs) - a).max() < 1e-2
    
def test_compute_pcs_3d():
    """Test PCA on a 3D array."""
    x1 = np.random.randn(20000, 2) * np.array([[10., 1.]])
    x2 = np.random.randn(20000, 2) * np.array([[1., 10.]])
    x = np.dstack((x1, x2))
    # Compute the PCs.
    pcs = compute_pcs(x)
    assert pcs.ndim == 3
    assert np.linalg.norm(pcs[0,:,0] - np.array([-1., 0.])) < 1e-2
    assert np.linalg.norm(pcs[1,:,0] - np.array([0., -1.])) < 1e-2
    assert np.linalg.norm(pcs[0,:,1] - np.array([0, 1.])) < 1e-2
    assert np.linalg.norm(pcs[1,:,1] - np.array([-1., 0.])) < 1e-2
    
def test_project_pcs():
    x1 = np.random.randn(20000, 2) * np.array([[10., 1.]])
    x2 = np.random.randn(20000, 2) * np.array([[1., 10.]])
    x = np.dstack((x1, x2))
    # Compute the PCs.
    pcs = compute_pcs(x)
    # Project the PCs.
    x_proj = project_pcs(x[0,...], pcs)
    assert x_proj.shape == (2, 2)
    
    