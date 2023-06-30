import pickle
import yaml
import wandb
import torch
import numpy as np
from param_parser import parameter_parser_train
from models.cmlp import cMLP, train_model_ista


def setup_wandb(cfg):
    wandb.init(
        project=cfg['wandb']['project'],
        name=cfg['wandb']['name'],
        config=cfg
    )
    # config = wandb.config
    return cfg


def main():
    args = parameter_parser_train()

    with open(args.yaml_path, "r") as stream: #parser
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # cfg = setup_wandb(cfg)
    model_cfg = cfg['model']
    device = torch.device(model_cfg['device'])
    data_catagory = args.data_catagory
    data_path = f'{args.data_path}/{data_catagory}.pickle'
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f'{data_catagory} data loaded...')
    X = data['Y']
    GC = data['GC']
    X = torch.tensor(X.T[np.newaxis], dtype=torch.float32, device=device)  # [1, 50, 30]
    
    cmlp = cMLP(X.shape[-1], lag=model_cfg['lag'], hidden=[model_cfg['hidden']]).cuda(device=device)

    train_loss_list = train_model_ista(
        cmlp = cmlp,
        X = X,
        lam = model_cfg['lam'],
        lam_ridge = model_cfg['lam_ridge'],
        lr = model_cfg['lr'],
        penalty = model_cfg['penalty'],
        max_iter = model_cfg['max_iter'],
        check_every = model_cfg['check_every']
    )

    torch.save(cmlp.state_dict(), f'{args.model_save_path}/cmlp_{args.data_catagory}.pt')


if __name__ == '__main__':
    main()


