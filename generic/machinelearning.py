import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from MATLAB import *

# %% ----- Curve fitting --------
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

# %% ----- K-means -------------
def elbow_curve(X, max_clusters=15, plot=False, *args, **kwargs):
    """
    Return the elbow curve for K-means clustering

    Example:

    from sklearn.datasets import make_blobs
    X, y_varied = make_blobs(n_samples=100,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=42)
    distortions, best_K = elbow_curve(X, max_clusters=10, plot=True)

    Inputs:
        * X: [n_samples, n_features]
        * max_clusters: max number of clusters to calculate. Default 15.
        * distance: distance metric used for clustering. Default 'euclidean'
        * *args, **kwargs: additional arguments for kmeans function

    Return:
        * distortions: arary of within-group sum of squares
        * best_k: best K value
    """
    # Calculate the elbow curve
    distortions = np.zeros(max_clusters)
    for k in range(0, max_clusters):
        D = KMeans(n_clusters=k+1).fit(X)
        for s in range(k+1):
            distortions[k] = distortions[k] + \
                np.sum((X[D.labels_==s, :] - D.cluster_centers_[s, :])**2)

    best_idx = find_point_closest(distortions)

    if plot:
        plt.plot(np.arange(1, max_clusters+1), distortions, '-o')
        plt.plot(best_idx+1, distortions[best_idx], 'ro')
        plt.xticks(np.arange(1, max_clusters+1))
        plt.xlabel('K')
        plt.ylabel('Distortions')

    return distortions, best_idx + 1 # return K


def find_point_closest(curve, plot=False):
    """
    Given an elbow curve, find the index of best K
    (then best K would be this index + 1)
    Can also be applied to ROC curve
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    nPoints = len(curve)
    allCoord = np.c_[np.arange(0, nPoints)+1, curve]    # SO formatting

    # pull out first point
    firstPoint = allCoord[0,:]

    # get vector between first and last point - this is the line
    lineVec = allCoord[-1,:] - firstPoint

    # normalize the line vector
    lineVecN = lineVec / np.sqrt(np.sum(lineVec**2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = vecFromFirst.dot(lineVecN)
    vecFromFirstParallel = np.matmul(scalarProduct[:, np.newaxis], lineVecN[np.newaxis, :])
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine**2,1));

    # plot the distance to the line
    # plt.plot(distToLine)

    # now all you need is to find the maximum
    idxOfBestPoint = np.argmax(distToLine)
    if plot:
        # plot
        plt.plot(curve)
        plt.plot(allCoord[idxOfBestPoint,0], allCoord[idxOfBestPoint,1], 'or')

    return idxOfBestPoint
