import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix, coo_matrix, dok_matrix, load_npz, save_npz
from tqdm import tqdm
from pdb import set_trace

def multiply_U_W(U, W, R):
    """Compute product of U.T and W, 
    but only at the position where R is available"""
    iU, iW = R.tocoo().row, R.tocoo().col #np.nonzero(R)
    #values = np.sum(U[iU, :] * W[iW, :], axis=1)
    values = np.einsum('ij,ij->i', U[iU, :], W[iW, :])
    result = coo_matrix((values, (iU, iW))).tocsc()
    return result

def R_plus_cv(R, cv):
    """Sparse matrix plus a column vector"""
    R = R.tocsc()
    R.data += np.take(cv, R.indices)
    return R

def R_plus_rv(R, rv):
    """Sparse matrix plus a row vector"""
    R = R.tocsr()
    R.data += np.take(rv, R.indices)
    return R

def R_op_cv(R, cv, op='plus'):
    """Sparse matrix (+-*/) a column vector"""
    R = R.tocsc()
    if op == 'plus':
        R.data += np.take(cv, R.indices)
    elif op == 'minus':
        R.data -= np.take(cv, R.indices)
    elif op == 'multiply':
        R.data *= np.take(cv, R.indices)
    elif op == 'divide':
        R.data /= np.take(cv, R.indices)
    else:
        raise(ValueError(f'Unknown operator {op}'))
    return R

def R_op_rv(R, rv, op='plus'):
    """Sparse matrix (+-*/) a row vector"""
    R = R.tocsr()    
    if op == 'plus':
        R.data += np.take(rv, R.indices)
    elif op == 'minus':
        R.data -= np.take(rv, R.indices)
    elif op == 'multiply':
        R.data *= np.take(rv, R.indices)
    elif op == 'divide':
        R.data /= np.take(rv, R.indices)
    else:
        raise(ValueError(f'Unknown operator {op}'))
    return R

def sparse_mean(R, axis=0):
    """Compute mean of a sparse matrix along a certain axis"""
    R = R.tocsc()
    # Compute mean
    R_mean = np.array(R.sum(axis=axis) / np.expand_dims(R.getnnz(axis=axis), axis=axis)) # dense, kept dim
    return R_mean

def sparse_var(R, axis=0):
    """Compute variance of a sparse matrix along a certain axis"""
    R = R.tocsc()
    # Compute mean
    R_mean = sparse_mean(R, axis=axis)
    # Compute variance
    R_var = np.array(R.multiply(R).sum(axis=axis) / np.expand_dims(R.getnnz(axis=axis), axis=axis)) - R_mean**2
    return R_var

def sparse_std(R, axis=0):
    """Compute standard deviation of a sparse matrix along a certain axis"""
    # Compute variance
    R_var = sparse_var(R, axis=axis)
    # Compute standard deviation
    R_std = np.sqrt(R_var)
    return R_std

def sparse_stats(R, axis=0):
    R = R.tocsc()
    R_mean = np.array(R.sum(axis=axis) / np.expand_dims(R.getnnz(axis=axis), axis=axis)) # dense, kept dim
    R_var = np.array(R.multiply(R).sum(axis=axis) / np.expand_dims(R.getnnz(axis=axis), axis=axis)) - R_mean**2
    R_std = np.sqrt(R_var)
    return R_mean, R_var, R_std

def sparse_symmetrify(R):
    R = R + R.T.multiply(R.T > R) - R.multiply(R.T > R)
    return R
    
def regularized_matrix_factorization(R, K=25, l2_reg=0.001, maxiter=100, random_state=42):
    """
    Parameters:
    --------
    * R : rating sparse matrix of size N x M
    * K: number of features to use
    * l2_reg: L2 regularization size
    * maxiter : max number of iterations
    * random_state: for testing purposes only
    Returns:
    --------
    U : N x K user matrix
    W : M x K item matrix
    b : length N vector of user bias
    c : length M vector of item bias
    mu: global mean of the user rating matrix
    """
    N, M = R.shape
    R = R.tocsc()
    # Calculate global mean mu
    mu = R.mean()
    
    # Initialize the user and item matrix
    rs = np.random.RandomState(random_state)
    U = rs.rand(N, K)
    W = rs.rand(M, K)
    U_W_prod = multiply_U_W(U, W, R) # sparse N x M
    
    # Initialize the bias terms
    b = rs.rand(N)
    c = rs.rand(M)
    
    # Counting cardinalities
    card_psi = R.getnnz(axis=1)
    card_omega = R.getnnz(axis=0)
    total_card = R.getnnz()
    
    # Initialize loss
    J = np.zeros(maxiter)
    
    # Iterate
    for epoch in tqdm(range(maxiter)):
        #print('epoch: ', epoch)
        # Compute prediction
        R_hat = U_W_prod.copy()
        R_hat.data += mu # add mu
        R_hat = R_plus_cv(R_hat, b) # add b, of length N
        R_hat = R_plus_rv(R_hat, c).tocsc() # add c, of length M
        
        # Compute regularized loss
        J[epoch] = ((R.data - R_hat.data)**2).sum() + l2_reg*(np.sum(U**2) + np.sum(W**2) + np.sum(b**2) + np.sum(c**2))
        #J[epoch] = np.sum((R.toarray() - R_hat.toarray())**2)
        
        J[epoch] = J[epoch] / total_card # mean squared error
        
        # Update the parameters
        R_b_c_mu = R.copy()
        R_b_c_mu.data -= mu
        R_b_c_mu = R_plus_cv(R_b_c_mu, -b)
        R_b_c_mu = R_plus_rv(R_b_c_mu, -c).tocsc()
        
        U = np.linalg.solve(W.T @ W + l2_reg * np.eye(K)[np.newaxis, :, :], 
                             R_b_c_mu @ W) # NxK = solve(1xKxK, NxK)
        W = np.linalg.solve(U.T @ U + l2_reg * np.eye(K)[np.newaxis, :, :], 
                             R_b_c_mu.T @ U) # MxK = solve(1xKxK, MxK) 
        U_W_prod = multiply_U_W(U, W, R)
        
        R_u_mu = R - U_W_prod
        R_u_mu.data -= mu
        R_u_c_mu = R_u_mu.copy()
        R_u_c_mu = R_plus_rv(R_u_c_mu, -c).tocsc()
        R_u_c_mu = np.asarray(R_u_c_mu.sum(axis=1)).flatten()
        R_u_b_mu = R_u_mu.copy()
        R_u_b_mu = R_plus_cv(R_u_b_mu, -b)
        R_u_b_mu = np.asarray(R_u_b_mu.sum(axis=0)).flatten()
        
        b = 1 / (card_psi   + l2_reg) * R_u_c_mu # length N
        c = 1 / (card_omega + l2_reg) * R_u_b_mu # legnth M
            
    return U, W, b, c, mu, J, R_hat

if __name__ == '__main__':
    tmp_A = np.array([[1,2,0], [0, 1, 3], [3, 2, 0], [1, 0, 5]], dtype=float)
    print(tmp_A)
    nan_A = tmp_A.copy()
    nan_A[nan_A<1E-6] = np.nan
    print(nan_A)
    A = scipy.sparse.csc_matrix(tmp_A)
    row_mean, row_var, row_std = sparse_stats(A, axis=0)
    print('sparse mean', row_mean)
    print('sparse var', row_var)
    print('sparse std', row_std)
    A_new = R_plus_rv(A, -row_mean.ravel())
    print('remove mean', A_new.toarray())
    A_new = R_op_rv(A_new, row_std.ravel(), op='divide')
    print('divide std', A_new.toarray())

    new_row_mean, _, new_row_std = sparse_stats(A_new, axis=0)
    print('sparse new mean:', new_row_mean)
    print('sparse new std:', new_row_std)

    print('numpy mean:', np.nanmean(nan_A, axis=0))
    print('numpy var:', np.nanvar(nan_A, axis=0))
    print('numpy std:', np.nanstd(nan_A, axis=0))
    SS = nan_A - np.nanmean(nan_A, axis=0, keepdims=True)
    print('remove_mean', SS)
    SS = SS/np.nanstd(nan_A, axis=0, keepdims=True)
    print('divide std', SS)
    print('numpy new mean: ', np.nanmean(SS, axis=0))
    print('numpy new std: ', np.nanstd(SS, axis=0))
