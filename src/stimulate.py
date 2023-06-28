import numpy as np


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

def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)           # (p, p)
    beta = np.eye(p) * beta_value       # (p, p)

    # choose non zero causality
    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])  # (p, p * lag)
    beta = make_var_stationary(beta)              # (p, p * lag)
    
    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))     # (p, T + burn_in)
    X = np.zeros((p, T + burn_in))                                 # (p, T + burn_in)
    
    for t in range(lag, T + burn_in):
        # (p, p * lag) dot (p * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def simulate_var_endogenous(k, S, lag, non_zeroes=[0], beta_value=0.4, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(k, dtype=int)           # (k, k)
    beta = np.eye(k) * beta_value       # (k, k)

    # add non zero causality
    for col in non_zeroes:
        beta[:, col] = beta_value
        GC[:, col] = 1
    
    beta = np.hstack([beta for _ in range(lag)])  # (k, k * lag)
    
    # Generate data.
    burn_in = int(0.1 * S)
    errors = np.random.normal(scale=sd, size=(k, S + burn_in))     # (k, S + burn_in)
    X = np.zeros((k, S + burn_in))                                 # (k, S + burn_in)
    
    for t in range(lag, S + burn_in):
        # (k, k * lag) dot (k * lag, )

        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t]
    
    data = {
        'GC': GC,
        'Y': X.T[burn_in:].T,
        'beta': beta,
    }

    return data


def simulate_var_latent(k, S, lag, non_zeroes=[0], beta_value=0.4, alpha_value=0.2, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(k, dtype=int)           # (k, k)
    beta = np.eye(k) * beta_value       # (k, k)
    alpha = np.eye(k) * alpha_value     # (k, k)

    # add non zero causality
    for col in non_zeroes:
        beta[:, col] = beta_value
        alpha[:, col] = alpha_value
        GC[:, col] = 1

    beta = np.hstack([beta for _ in range(lag)])    # (k, k * lag)

    # Generate data.
    burn_in = int(0.1 * S)
    errors = np.random.normal(scale=sd, size=(k, S + burn_in))        # (k, S + burn_in)
    X = np.zeros((k, S + burn_in))                                    # (k, S + burn_in)
    L = np.random.binomial(1, 0.5, (k, S + burn_in)).astype(float)    # (k, S + burn_in)
    
    for t in range(lag, S + burn_in):
        # (k, k) dot (k, )
        L[:, t] = np.dot(alpha, L[:, t])
        # (k, k * lag) dot (k * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] = X[:, t] + L[:, t] + errors[:, t]

    data = {
        'GC': GC,
        'Y': X.T[burn_in:].T,
        'beta': beta,
        'X': L.T[burn_in:].T,
        'alpha': alpha
    }

    return data


def simulate_var_retail_latent(k, S, lag, beta_value=0.8, cross_cat_value=0.2, latent_value=0.2, sd=0.1, seed=0):
    def generate_random_gc(k):
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
   
    if seed is not None:
        np.random.seed(seed)

    burn_in = int(0.1 * S)
    theta_sign = np.random.binomial(1, 0.5, 4)
    
    # Set up coefficients and Granger causality ground truth (k, k).
    GC = np.eye(k, dtype=int)
    beta = np.eye(k) * beta_value if theta_sign[0] else np.eye(k) * (-beta_value)
    alpha = np.eye(k) * latent_value if theta_sign[1] else np.eye(k) * (-latent_value)
    gamma = np.eye(k) * latent_value if theta_sign[2] else np.eye(k) * (-latent_value)
    sigma = np.eye(k) * latent_value if theta_sign[3] else np.eye(k) * (-latent_value)

    # add non zero causality
    gc_sign = np.random.binomial(1, 0.5, k)
    gc = generate_random_gc(k)

    for i, sign_num in enumerate(zip(gc_sign, gc)):
        sign, num = sign_num
        beta[i, num] = cross_cat_value if sign else (-cross_cat_value)
        GC[i, num] = 1

    beta = np.hstack([beta for _ in range(lag)])        # (k, k * lag)

    # Generate data. (k, S + burn_in)
    X = np.zeros((k, S + burn_in))                                               
    PZC = np.random.triangular(-0.1, 0, 0.01, (k, S + burn_in))
    AD = np.random.beta(1.0, 60.0, (k, S + burn_in))
    DI = np.random.beta(2.0, 25.0, (k, S + burn_in))
    
    for t in range(lag, S + burn_in):
        # (k, k) dot (k, )
        PZC[:, t] = np.dot(alpha, PZC[:, t])
        AD[:, t] = np.dot(gamma, AD[:, t])
        DI[:, t] = np.dot(sigma, DI[:, t])

        # (k, k * lag) dot (k * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] = X[:, t] + PZC[:, t] + AD[:, t] + DI[:, t]
    
    data = {
        'GC': GC,
        'Y': X.T[burn_in:].T,
        'beta': beta,
        'PZC': PZC.T[burn_in:].T,
        'alpha': alpha,
        'AD': AD.T[burn_in:].T,
        'gamma': gamma,
        'DI': DI.T[burn_in:].T,
        'sigma': sigma
    }

    return data


simulate_var_endogenous(k=30, S=50, lag=1)
simulate_var_latent(k=30, S=50, lag=1)
simulate_var_retail_latent(k=30, S=50, lag=1)