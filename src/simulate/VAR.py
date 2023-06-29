import numpy as np
import simulate.utils as utils


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
    beta = utils.make_var_stationary(beta)              # (p, p * lag)
    
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


def simulate_var_retail_latent(k, S, lag, cross_cat_value=0.2, seed=0):
    if seed is not None:
        np.random.seed(seed)

    burn_in = int(0.1 * S)
    theta_sign = np.random.binomial(1, 0.5, 4)
    
    # Set up coefficients and Granger causality ground truth (k, k).
    GC = np.eye(k, dtype=int)
    beta, GC = utils.coef_sim(30, 60/900, [-0.25, 0.25], -0.5, GC=GC, seed=0)
    alpha, _ = utils.coef_sim(30, 60/900, [-0.15, 0.15], -0.3, seed=10)
    gamma, _ = utils.coef_sim(30, 40/900, [-0.15, 0.15], -0.3, seed=100)
    delta, _ = utils.coef_sim(30, 40/900, [-0.15, 0.15], -0.3, seed=1000)

    beta = np.hstack([beta for _ in range(lag)])        # (k, k * lag)

    # Generate data. (k, S + burn_in)
    X = np.zeros((k, S + burn_in))                                               
    PZC = np.random.triangular(-0.1, 0, 0.01, (k, S + burn_in))
    AD = np.random.beta(0.5, 26.0, (k, S + burn_in))
    DI = np.random.beta(2.5, 34.0, (k, S + burn_in))
    sigma_out = utils.sigma_sim(k, [0.2]*k, seed=seed)
    SIGMA = sigma_out['sparse']
    ECt = np.random.multivariate_normal(np.zeros(k), SIGMA, size=(S + burn_in)).T
    
    for t in range(lag, S + burn_in):
        # (k, k) dot (k, )
        PZC[:, t] = np.dot(alpha, PZC[:, t])
        AD[:, t] = np.dot(gamma, AD[:, t])
        DI[:, t] = np.dot(delta, DI[:, t])

        # (k, k * lag) dot (k * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] = X[:, t] + PZC[:, t] + AD[:, t] + DI[:, t] + ECt[:, t]
    
    data = {
        'GC': GC,
        'Y': X.T[burn_in:].T,
        'beta': beta,
        'PZC': PZC.T[burn_in:].T,
        'alpha': alpha,
        'AD': AD.T[burn_in:].T,
        'gamma': gamma,
        'DI': DI.T[burn_in:].T,
        'delta': delta,
        'sigma': SIGMA,
        'ECt': ECt
    }
    
    return data


# simulate_var_endogenous(k=30, S=50, lag=1)
# simulate_var_latent(k=30, S=50, lag=1)
# simulate_var_retail_latent(k=30, S=50, lag=1)