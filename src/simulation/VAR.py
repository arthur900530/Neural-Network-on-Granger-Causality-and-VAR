import numpy as np
import simulation.utils as utils
import copy


# orig paper Neural Cranger Causality
def simulate_var(k, S, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(k, dtype=int)           # (p, p)
    beta = np.eye(k) * beta_value       # (p, p)

    # choose non zero causality
    num_nonzero = int(k * sparsity) - 1
    for i in range(k):
        choice = np.random.choice(k - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])  # (p, p * lag)
    beta = utils.make_var_stationary(beta)              # (p, p * lag)
    
    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(k, S + burn_in))     # (p, T + burn_in)
    X = np.zeros((k, S + burn_in))                                 # (p, T + burn_in)
    
    for t in range(lag, S + burn_in):
        # (p, p * lag) dot (p * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def simulate_var_endogenous(cfg):
    k = cfg['k']
    time_span = cfg['T']
    lag = cfg['lag']
    non_zeroes = cfg['non_zeroes']
    beta_value = cfg['beta_value']
    error_sd = cfg['error_sd']
    seed = cfg['seed']

    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth. (k, k)
    GC = np.eye(k, dtype=int)           
    beta = np.eye(k) * beta_value       

    # add non zero causality
    for col in non_zeroes:
        beta[:, col] = beta_value
        GC[:, col] = 1
    
    beta = np.hstack([beta for _ in range(lag)])  # (k, k * lag)
    
    # Generate data. (k, S + burn_in)
    burn_in = int(0.1 * time_span)
    errors = np.random.normal(scale=error_sd, size=(k, 2 * time_span + burn_in))  
    X = np.zeros((k, 2 * time_span + burn_in))                                    
    
    for t in range(lag, 2 * time_span + burn_in):
        # (k, k * lag) dot (k * lag, )
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t]

    X = X.T[burn_in:].T
    X_train = X[:, :time_span]
    X_val = X[:, time_span:]

    data = {
        'GC': GC,
        'Y': X_train,
        'Y_val': X_val,
        'beta': beta,
    }
    return data


def simulate_var_latent(cfg):
    k = cfg['k']
    time_span = cfg['T']
    lag = cfg['lag']
    non_zeroes = cfg['non_zeroes']
    beta_value = cfg['beta_value']
    alpha_value = cfg['alpha_value']
    error_sd = cfg['error_sd']
    seed = cfg['seed']
    
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth. (k, k)
    GC = np.eye(k, dtype=int)                
    beta = np.eye(k) * beta_value
    alpha = np.eye(k) * alpha_value

    # add non zero causality
    for col in non_zeroes:
        beta[:, col] = beta_value
        alpha[:, col] = alpha_value
        GC[:, col] = 1

    beta = np.hstack([beta for _ in range(lag)])  # (k, k * lag)

    # Generate data. (k, S + burn_in)
    burn_in = int(0.1 * time_span)
    errors = np.random.normal(scale=error_sd, size=(k, 2 * time_span + burn_in)) 
    X = np.zeros((k, 2 * time_span + burn_in))                          
    orig_X = np.zeros((k, 2 * time_span + burn_in))
    latent = np.random.binomial(1, 0.5, (k, 2 * time_span + burn_in)).astype(float)
    
    for t in range(lag, 2 * time_span + burn_in):
        latent[:, t] = np.dot(alpha, latent[:, t])                            # (k, k) dot (k,)
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))  # (k, k * lag) dot (k * lag,)
        orig_X[:, t] = np.dot(beta, orig_X[:, (t-lag):t].flatten(order='F')) + errors[:, t]
        X[:, t] = X[:, t] + latent[:, t] + errors[:, t]

    X = X.T[burn_in:].T
    latent = latent.T[burn_in:].T
    X_train = X[:, :time_span]
    X_val = X[:, time_span:]

    orig_X = orig_X.T[burn_in:].T
    orig_X_train = orig_X[:, :time_span]
    orig_X_val = orig_X[:, time_span:]
    latent_train = latent[:, :time_span]
    latent_val = latent[:, time_span:]
    data = {
        'GC': GC,
        'Y': X_train,
        'Y_val': X_val,
        'beta': beta,
        'alpha': alpha,
        'latent_train': latent_train,
        'latent_val': latent_val,
        'orig_X_train': orig_X_train,
        'orig_X_val': orig_X_val,
    }
    return data


def simulate_var_retail_latent(cfg):
    k = cfg['k']
    time_span = cfg['T']
    lag = cfg['lag']
    seed = cfg['seed']

    if seed is not None:
        np.random.seed(seed)

    burn_in = int(0.1 * time_span)
    
    # Set up coefficients and Granger causality ground truth. (k, k)
    GC = np.eye(k, dtype=int)
    beta, GC = utils.coef_sim(k, cfg['beta'], GC=GC)
    alpha, _ = utils.coef_sim(k, cfg['alpha'])
    gamma, _ = utils.coef_sim(k, cfg['gamma'])
    delta, _ = utils.coef_sim(k, cfg['delta'])

    beta = np.hstack([beta for _ in range(lag)])  # (k, k * lag)

    # Generate data. (k, S + burn_in)
    X = np.zeros((k, 2 * time_span + burn_in))                                               
    PZC = np.random.triangular(-0.1, 0, 0.01, (k, 2 * time_span + burn_in))
    AD = np.random.beta(0.5, 26.0, (k, 2 * time_span + burn_in))
    DI = np.random.beta(2.5, 34.0, (k, 2 * time_span + burn_in))
    sigma_out = utils.sigma_sim(k, [0.2] * k, seed=seed)
    SIGMA = sigma_out['sparse']
    ECt = np.random.multivariate_normal(np.zeros(k), SIGMA, size=(2 * time_span + burn_in)).T
    
    for t in range(lag, 2 * time_span + burn_in):
        PZC[:, t] = np.dot(alpha, PZC[:, t])                        # (k, k) dot (k, )
        AD[:, t] = np.dot(gamma, AD[:, t])
        DI[:, t] = np.dot(delta, DI[:, t])
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))  # (k, k * lag) dot (k * lag, )
        X[:, t] = X[:, t] + PZC[:, t] + AD[:, t] + DI[:, t] + ECt[:, t]
    
    X = X.T[burn_in:].T
    X_train = X[:, :time_span]
    X_val = X[:, time_span:]

    data = {
        'GC': GC,
        'Y': X_train,
        'Y_val': X_val,
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
