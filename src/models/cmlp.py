import torch
import torch.nn as nn
from models.model_helper import activation_helper


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation, prime):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]
        if not prime:
            for d_in, d_out in zip(hidden, hidden[1:] + [1]):
                layer = nn.Conv1d(d_in, d_out, 1)
                modules.append(layer)
        else:
            layer = nn.Linear(hidden[0], 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)               # Batch, p, T
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)                       # Batch, hidden, T

        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation, prime=False):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation, prime)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True, prime=False):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if not prime:
            if ignore_lag:
                GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                    for net in self.networks]
            else:
                GC = [torch.norm(net.layers[0].weight, dim=0)
                    for net in self.networks]
        else:
            GC = []
            for net in self.networks:
                if net.layers[0].weight.shape[-1] > 1:
                    l0 = torch.norm(net.layers[0].weight, dim=(2))
                else:
                    l0 = torch.squeeze(net.layers[0].weight, -1)
                gc = torch.squeeze(torch.matmul(net.layers[1].weight, l0), 0)
                GC.append(gc)

        GC = torch.stack(GC)
        if threshold:
            GC = (GC != 0).int()
            
        return GC


class cMLPSparse(nn.Module):
    def __init__(self, num_series, sparsity, lag, hidden, activation):
        '''
        cMLP model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPSparse, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(MLP(num_inputs, lag, hidden, activation))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([self.networks[i](X[:, :, self.sparsity[i]])
                          for i in range(self.p)], dim=2)


class composedMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation, cmlp_count):
        super(composedMLP, self).__init__()
        self.cmlp_count = cmlp_count
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.cmlps = nn.ModuleList([
            cMLP(num_series, lag, hidden, activation)
            for _ in range(cmlp_count)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        x = torch.zeros(X.shape)
        for cmlp in self.cmlps:
            x = torch.add(x, cmlp(X))

    def GC(self, threshold=True, ignore_lag=True):
        GC_list = []
        for cmlp in self.cmlps:
            if ignore_lag:
                GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                    for net in cmlp.networks]
            else:
                GC = [torch.norm(net.layers[0].weight, dim=0)
                    for net in cmlp.networks]
            GC = torch.stack(GC)
            if threshold:
                GC = (GC != 0).int()
            GC_list.append(GC)
        return GC_list