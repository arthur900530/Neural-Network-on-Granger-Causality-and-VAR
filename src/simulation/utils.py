import numpy as np
from fractions import Fraction


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''                      # beta: 10, 30
    p = beta.shape[0]                                                            # 10
    lag = beta.shape[1] // p                                                     # 3
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))    # 20, 30
    beta_tilde = np.vstack((beta, bottom))                                       # 30, 30
    eigvals = np.linalg.eigvals(beta_tilde)                                      # 30, 
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta
    
def generate_random_correlation(k):
        gc = np.array([])
        for i in range(k):
            while True:
                c = np.random.randint(0, k, 1)
                if c == i:
                    continue
                else:
                    gc = np.append(gc, c)
                    break
        return gc.astype(int)

def coef_sim(k, cfg, GC=None):
    """
    Generate a coefficient matrix with specified properties.
    
    Args:
    - q: dimension of the coefficient matrix.
    - sparsity: number of non-zero values within the full matrix.
    - coeff_lim: limits for positive/negative values in the coefficient matrix.
    - diag: value of the diagonal (own) effects.
    - seed: seed value for random number generation (default: 8888).
    
    Returns:
    - beta: generated coefficient matrix.
    """
    sparsity = float(Fraction(cfg['sparsity']))
    coeff_lim = cfg['coeff_lim']
    diag = cfg['diag']
    seed = cfg['seed']

    np.random.seed(seed)

    if sparsity < (1/k):
        raise ValueError("Sparsity error. Should be greater than 1/q")
    
    beta = np.zeros((k, k))
    np.fill_diagonal(beta, diag)
    
    total_values = int(max(1, k * k * sparsity - k))
    valued_indices = np.random.choice(np.arange(1, k * k - k), total_values, replace=False)

    up_tri = np.triu_indices(k, k=1)
    low_tri = np.tril_indices(k, k=-1)
    
    indices = list(zip(up_tri[0], up_tri[1])) + list(zip(low_tri[0], low_tri[1]))
    indices = np.array([list(row) for row in indices])

    for i in indices[valued_indices]:
        beta[i[0]][i[1]] = np.random.choice(coeff_lim)
        if isinstance(GC, type(np.array([]))):
            GC[i[0]][i[1]] = 1
    
    return beta, GC

def sigma_sim(k, sd_vec, seed=0):
    np.random.seed(seed)
    
    cor = {}
    cov_raw = np.outer(sd_vec, sd_vec)
    
    # Diagonal: Identity correlations
    diagonal = np.eye(k)
    cov_diagonal = diagonal * cov_raw
    cor['diagonal'] = cov_diagonal / np.sqrt(np.outer(sd_vec, sd_vec))
    
    # Sparse: Randomly picked q cross-correlation = 0.3
    sparse = diagonal.copy()
    indices = np.tril_indices(k, -1)
    random_indices = np.random.choice(len(indices[0]), size=k, replace=False)
    sparse[indices[0][random_indices], indices[1][random_indices]] = np.random.choice([-0.3, 0.3], size=k)
    cov_sparse = np.dot(sparse, sparse.T) * cov_raw
    cor['sparse'] = cov_sparse / np.sqrt(np.outer(sd_vec, sd_vec))
    
    s = round(0.8 * k * (k - 1) / 2)
    # Dense: Randomly picked s cross-correlation = 0.3
    dense = diagonal.copy()
    random_indices = np.random.choice(len(indices[0]), size=s, replace=False)
    dense[indices[0][random_indices], indices[1][random_indices]] = np.random.choice([-0.3, 0.3], size=s)
    cov_dense = np.dot(dense, dense.T) * cov_raw
    cor['dense'] = cov_dense / np.sqrt(np.outer(sd_vec, sd_vec))
    
    out = {'diagonal': cov_diagonal,
           'sparse': cov_sparse,
           'dense': cov_dense,
           'cor': cor,
           'sigma': sd_vec}
    
    return out