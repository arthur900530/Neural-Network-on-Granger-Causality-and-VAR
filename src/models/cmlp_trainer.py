import yaml
import torch
import torch.nn as nn
import pickle
import numpy as np
from models.cmlp import cMLP
from copy import deepcopy
from tqdm import trange


class CMLP_Container():
    def __init__(self, args=None, eval_config=None) -> None:
        if args:
            self.args = args
            self.__setup_configuration()
            self.__load_data()
            self.__setup_model()
        elif eval_config:
            self.eval_config = eval_config
            self.__setup_eval_configuration()
            self.__load_data()
            self.__load_model()
            
    def __setup_eval_configuration(self):
        with open(self.eval_config['yaml_path'], "r") as stream:
            cfg = yaml.safe_load(stream)
        self.model_cfg = cfg['model']
        self.device = torch.device(self.model_cfg['device'])
        self.data_catagory = self.eval_config['catagory']
        self.data_path = f'{self.eval_config["data_dir"]}/{self.data_catagory}.pickle'
    
    def __setup_configuration(self):
        with open(self.args.yaml_path, "r") as stream:
            cfg = yaml.safe_load(stream)
        self.model_cfg = cfg['model']
        self.device = torch.device(self.model_cfg['device'])
        self.data_catagory = self.args.catagory
        self.data_path = f'{self.args.data_dir}/{self.data_catagory}.pickle'
        self.train_loss_list = None
        self.train_mse_list = None

    def __load_data(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
            print(f'{self.data_catagory} data loaded...')
        X = self.data['Y']
        self.GC = self.data['GC']
        self.beta = self.data['beta']
        self.X = torch.tensor(X.T[np.newaxis], dtype=torch.float32, device=self.device)  # [1, 50, 30]
    
    def __setup_model(self):
        self.cmlp = cMLP(
              self.X.shape[-1],
              lag=self.model_cfg['lag'],
              hidden=[self.model_cfg['hidden']],
              activation=self.model_cfg['activation']
        ).cuda(device=self.device)
    
    def reset_model(self):
        self.__setup_model()
        self.train_loss_list = None
        self.train_mse_list = None
    
    def __load_model(self):
        self.__setup_model()
        self.cmlp.load_state_dict(torch.load(f'{self.eval_config["model_save_dir"]}/cmlp_{self.data_catagory}.pt'))
    
    def save_model_and_loss(self):
        torch.save(self.cmlp.state_dict(), f'{self.args.model_save_dir}/cmlp_{self.data_catagory}.pt')
        if self.train_loss_list:
            with open(f"{self.args.model_save_dir}/cmlp_{self.data_catagory}_loss.pickle", 'wb') as f:
                pickle.dump(self.train_loss_list, f)
        if self.train_mse_list:
            with open(f"{self.args.model_save_dir}/cmlp_{self.data_catagory}_mse_loss.pickle", 'wb') as f:
                pickle.dump(self.train_mse_list, f)
        
        print(f'Model and loss saved at {self.args.model_save_dir}')

    def __prox_update(self, network, lam, lr, penalty):
        '''
        Perform in place proximal update on first layer weight matrix.

        Args:
        network: MLP network.
        lam: regularization parameter.
        lr: learning rate.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W = network.layers[0].weight
        hidden, p, lag = W.shape
        if penalty == 'GL':
            norm = torch.norm(W, dim=(0, 2), keepdim=True)
            W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
        elif penalty == 'GSGL':
            norm = torch.norm(W, dim=0, keepdim=True)
            W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
            norm = torch.norm(W, dim=(0, 2), keepdim=True)
            W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            for i in range(lag):
                norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
                W.data[:, :, :(i+1)] = (
                    (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
        else:
            raise ValueError('unsupported penalty: %s' % penalty)

    def __regularize(self, network, lam, penalty):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W = network.layers[0].weight
        hidden, p, lag = W.shape
        if penalty == 'GL':
            return lam * torch.sum(torch.norm(W, dim=(0, 2)))
        elif penalty == 'GSGL':
            return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                        + torch.sum(torch.norm(W, dim=0)))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                            for i in range(lag)])
        else:
            raise ValueError('unsupported penalty: %s' % penalty)

    def __ridge_regularize(self, network, lam):
        '''Apply ridge penalty at all subsequent layers.'''
        return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])

    def __restore_parameters(self, model, best_model):
        '''Move parameter values from best_model to model.'''
        for params, best_params in zip(model.parameters(), best_model.parameters()):
            params.data = best_params
        return model


    def train_model_gista(self, r=0.8, lr_min=1e-8, sigma=0.5,
                        monotone=False, m=10, lr_decay=0.5,
                        begin_line_search=True, switch_tol=1e-3, verbose=1):
        '''
        Train cMLP model with GISTA.

        Args:
        clstm: clstm model.
        X: tensor of data, shape (batch, T, p).
        lam: parameter for nonsmooth regularization.
        lam_ridge: parameter for ridge regularization on output layer.
        lr: learning rate.
        penalty: type of nonsmooth regularization.
        max_iter: max number of GISTA iterations.
        check_every: how frequently to record loss.
        r: for line search.
        lr_min: for line search.
        sigma: for line search.
        monotone: for line search.
        m: for line search.
        lr_decay: for adjusting initial learning rate of line search.
        begin_line_search: whether to begin with line search.
        switch_tol: tolerance for switching to line search.
        verbose: level of verbosity (0, 1, 2).
        '''
        cmlp = self.cmlp
        X = self.X
        lam = self.model_cfg['lam']
        lam_ridge = self.model_cfg['lam_ridge']
        lr = self.model_cfg['lr']
        penalty = self.model_cfg['penalty']
        max_iter = self.model_cfg['max_iter']
        check_every = self.model_cfg['check_every']

        p = cmlp.p
        lag = cmlp.lag
        cmlp_copy = deepcopy(cmlp)
        loss_fn = nn.MSELoss(reduction='mean')
        lr_list = [lr for _ in range(p)]

        # Calculate full loss.
        mse_list = []
        smooth_list = []
        loss_list = []
        for i in range(p):
            net = cmlp.networks[i]
            mse = loss_fn(net(X[:, :-1]), X[:, lag:, i:i+1])
            ridge = self.__ridge_regularize(net, lam_ridge)
            smooth = mse + ridge
            mse_list.append(mse)
            smooth_list.append(smooth)
            with torch.no_grad():
                nonsmooth = self.__regularize(net, lam, penalty)
                loss = smooth + nonsmooth
                loss_list.append(loss)

        # Set up lists for loss and mse.
        with torch.no_grad():
            loss_mean = sum(loss_list) / p
            mse_mean = sum(mse_list) / p
        train_loss_list = [loss_mean]
        train_mse_list = [mse_mean]

        # For switching to line search.
        line_search = begin_line_search

        # For line search criterion.
        done = [False for _ in range(p)]
        assert 0 < sigma <= 1
        assert m > 0
        if not monotone:
            last_losses = [[loss_list[i]] for i in range(p)]

        pbar = trange(max_iter)

        for it in pbar:
            # Backpropagate errors.
            sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

            # For next iteration.
            new_mse_list = []
            new_smooth_list = []
            new_loss_list = []

            # Perform GISTA step for each network.
            for i in range(p):
                # Skip if network converged.
                if done[i]:
                    new_mse_list.append(mse_list[i])
                    new_smooth_list.append(smooth_list[i])
                    new_loss_list.append(loss_list[i])
                    continue

                # Prepare for line search.
                step = False
                lr_it = lr_list[i]
                net = cmlp.networks[i]
                net_copy = cmlp_copy.networks[i]

                while not step:
                    # Perform tentative ISTA step.
                    for param, temp_param in zip(net.parameters(),
                                                net_copy.parameters()):
                        temp_param.data = param - lr_it * param.grad

                    # Proximal update.
                    self.__prox_update(net_copy, lam, lr_it, penalty)

                    # Check line search criterion.
                    mse = loss_fn(net_copy(X[:, :-1]), X[:, lag:, i:i+1])
                    ridge = self.__ridge_regularize(net_copy, lam_ridge)
                    smooth = mse + ridge
                    with torch.no_grad():
                        nonsmooth = self.__regularize(net_copy, lam, penalty)
                        loss = smooth + nonsmooth
                        tol = (0.5 * sigma / lr_it) * sum(
                            [torch.sum((param - temp_param) ** 2)
                            for param, temp_param in
                            zip(net.parameters(), net_copy.parameters())])

                    comp = loss_list[i] if monotone else max(last_losses[i])
                    if not line_search or (comp - loss) > tol:
                        step = True
                        if verbose > 1:
                            print('Taking step, network i = %d, lr = %f'
                                % (i, lr_it))
                            print('Gap = %f, tol = %f' % (comp - loss, tol))

                        # For next iteration.
                        new_mse_list.append(mse)
                        new_smooth_list.append(smooth)
                        new_loss_list.append(loss)

                        # Adjust initial learning rate.
                        lr_list[i] = (
                            (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                        if not monotone:
                            if len(last_losses[i]) == m:
                                last_losses[i].pop(0)
                            last_losses[i].append(loss)
                    else:
                        # Reduce learning rate.
                        lr_it *= r
                        if lr_it < lr_min:
                            done[i] = True
                            new_mse_list.append(mse_list[i])
                            new_smooth_list.append(smooth_list[i])
                            new_loss_list.append(loss_list[i])
                            if verbose > 0:
                                print('Network %d converged' % (i + 1))
                            break

                # Clean up.
                net.zero_grad()

                if step:
                    # Swap network parameters.
                    cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

            # For next iteration.
            mse_list = new_mse_list
            smooth_list = new_smooth_list
            loss_list = new_loss_list

            # Check if all networks have converged.
            if sum(done) == p:
                if verbose > 0:
                    print('Done at iteration = %d' % (it + 1))
                break

            # Check progress.
            if (it + 1) % check_every == 0:
                with torch.no_grad():
                    loss_mean = sum(loss_list) / p
                    mse_mean = sum(mse_list) / p
                    ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                    nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

                train_loss_list.append(loss_mean)
                train_mse_list.append(mse_mean)
                variable_usage = round((100 * torch.mean(cmlp.GC().float())).item(), 2)
                
                pbar.set_description(desc=f'Iter: {it+1}, Total loss: {round(loss_mean.item(), 5)}, MSE: {round(mse_mean.item(), 5)}, Ridge: {round(ridge_mean.item(), 5)}, Nonsmooth: {round(nonsmooth_mean.item(), 5)}, Variable usage: {variable_usage}%')

                # Check whether loss has increased.
                if not line_search:
                    if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                        line_search = True
                        if verbose > 0:
                            print('Switching to line search')
        self.cmlp = cmlp
        self.train_loss_list = train_loss_list
        self.train_mse_list = train_mse_list


    def train_model_adam(self, lookback=5, verbose=1):
        '''Train model with Adam.'''
        cmlp = self.cmlp
        X = self.X
        lam = self.model_cfg['lam']
        lam_ridge = self.model_cfg['lam_ridge']
        lr = self.model_cfg['lr']
        penalty = self.model_cfg['penalty']
        max_iter = self.model_cfg['max_iter']
        check_every = self.model_cfg['check_every']

        lag = cmlp.lag
        p = X.shape[-1]
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
        train_loss_list = []

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        pbar = trange(max_iter)

        for it in pbar:
            # Calculate loss.
            loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                        for i in range(p)])

            # Add penalty terms.
            if lam > 0:
                loss = loss + sum([self.regularize(net, lam, penalty)
                                for net in cmlp.networks])
            if lam_ridge > 0:
                loss = loss + sum([self.ridge_regularize(net, lam_ridge)
                                for net in cmlp.networks])

            # Take gradient step.
            loss.backward()
            optimizer.step()
            cmlp.zero_grad()

            # Check progress.
            if (it + 1) % check_every == 0:
                mean_loss = loss / p
                train_loss_list.append(mean_loss.detach())
                variable_usage = round((100 * torch.mean(cmlp.GC().float())).item(), 2)
                pbar.set_description(desc=f'Iter: {it+1}, Loss: {round(mean_loss.item(), 5)}, Variable usage: {variable_usage}%')

                # Check for early stopping.
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_it = it
                    best_model = deepcopy(cmlp)
                elif (it - best_it) == lookback * check_every:
                    if verbose:
                        print('Stopping early')
                    break

        # Restore best model.
        cmlp = self.__restore_parameters(cmlp, best_model)
        self.cmlp = cmlp

        self.train_loss_list = train_loss_list


    def train_model_ista(self, lookback=5, verbose=1):
        '''Train model with Adam.'''
        cmlp = self.cmlp
        X = self.X
        lam = self.model_cfg['lam']
        lam_ridge = self.model_cfg['lam_ridge']
        lr = self.model_cfg['lr']
        penalty = self.model_cfg['penalty']
        max_iter = self.model_cfg['max_iter']
        check_every = self.model_cfg['check_every']

        lag = cmlp.lag
        p = X.shape[-1]
        loss_fn = nn.MSELoss(reduction='mean')
        train_loss_list = []

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        # Calculate smooth error.
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])
        ridge = sum([self.__ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss + ridge

        pbar = trange(max_iter)

        for it in pbar:
            # Take gradient step.
            smooth.backward()
            for param in cmlp.parameters():
                param.data = param - lr * param.grad

            # Take prox step.
            if lam > 0:
                for net in cmlp.networks:
                    self.__prox_update(net, lam, lr, penalty)

            cmlp.zero_grad()

            # Calculate loss for next iteration.
            loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                        for i in range(p)])
            ridge = sum([self.__ridge_regularize(net, lam_ridge) for net in cmlp.networks])
            smooth = loss + ridge

            # Check progress.
            if (it + 1) % check_every == 0:
                # Add nonsmooth penalty.
                nonsmooth = sum([self.__regularize(net, lam, penalty)
                                for net in cmlp.networks])
                mean_loss = (smooth + nonsmooth) / p
                train_loss_list.append(mean_loss.detach())
                variable_usage = round((100 * torch.mean(cmlp.GC().float())).item(), 2)
                
                pbar.set_description(desc=f'Iter: {it+1}, Loss: {round(mean_loss.item(), 5)}, Variable usage: {variable_usage}%')

                # Check for early stopping.
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_it = it
                    best_model = deepcopy(cmlp)
                elif (it - best_it) == lookback * check_every:
                    if verbose:
                        print('Stopping early')
                    break

        # Restore best model.
        cmlp = self.__restore_parameters(cmlp, best_model)
        self.cmlp = cmlp

        self.train_loss_list = train_loss_list


    def train_unregularized(self, lookback=5, verbose=1):
        '''Train model with Adam and no regularization.'''
        cmlp = self.cmlp
        X = self.X
        lr = self.model_cfg['lr']
        max_iter = self.model_cfg['max_iter']
        check_every = self.model_cfg['check_every']

        lag = cmlp.lag
        p = X.shape[-1]
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
        train_loss_list = []

        # For early stopping.
        best_it = None
        best_loss = np.inf
        best_model = None

        pbar = trange(max_iter)

        for it in pbar:
            # Calculate loss.
            pred = cmlp(X[:, :-1])
            loss = sum([loss_fn(pred[:, :, i], X[:, lag:, i]) for i in range(p)])

            # Take gradient step.
            loss.backward()
            optimizer.step()
            cmlp.zero_grad()

            # Check progress.
            if (it + 1) % check_every == 0:
                mean_loss = loss / p
                train_loss_list.append(mean_loss.detach())
                variable_usage = round((100 * torch.mean(cmlp.GC().float())).item(), 2)

                pbar.set_description(desc=f'Iter: {it+1}, Loss: {round(mean_loss.item(), 5)}, Variable usage: {variable_usage}%')

                # Check for early stopping.
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_it = it
                    best_model = deepcopy(cmlp)
                elif (it - best_it) == lookback * check_every:
                    if verbose:
                        print('Stopping early')
                    break

        # Restore best model.
        cmlp = self.__restore_parameters(cmlp, best_model)
        self.cmlp = cmlp

        self.train_loss_list = train_loss_list


    def evaluate(self):
        theta = self.cmlp.GC(threshold=False).cpu().data.numpy()
        self.TPR = self.__TPR(theta)
        self.TNR = self.__TNR(theta)
        self.MAEE = self.__MAEE(theta)
        return f'TPR: {round(self.TPR, 3)}\nTNR: {round(self.TNR, 3)}\nMAEE: {round(self.MAEE, 3)}'

    
    def __TPR(self, theta):
        beta = self.beta
        num_pos = np.sum(np.greater(theta, 0) & np.greater(beta, 0))
        num_neg = np.sum(np.less(theta, 0) & np.less(beta, 0))
        numerator = num_pos + num_neg
        denominator = np.sum(np.not_equal(beta, 0))
        TNR = numerator / denominator
        return TNR

    def __TNR(self, theta):
        beta = self.beta
        numerator = np.sum(np.equal(theta, 0) & np.equal(beta, 0))
        denominator = np.sum(np.equal(beta, 0))
        TNR = numerator / denominator
        return TNR
        
    def __MAEE(self, theta):
        beta = self.beta
        norm = np.linalg.norm(theta - beta)
        MAEE = norm / beta.size
        return MAEE