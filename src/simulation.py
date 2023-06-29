import yaml
from simulation import VAR
import pickle


def main():
    with open("../configs/simulation.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    data_A = VAR.simulate_var_endogenous(cfg['endogenous'])
    data_B = VAR.simulate_var_latent(cfg['latent'])
    data_C = VAR.simulate_var_retail_latent(cfg['retail_latent'])

    with open(cfg['save_path']['A'], 'wb') as f:
        pickle.dump(data_A, f)
    print('Data A simulated and saved...')

    with open(cfg['save_path']['B'], 'wb') as f:
        pickle.dump(data_B, f)
    print('Data B simulated and saved...')

    with open(cfg['save_path']['C'], 'wb') as f:
        pickle.dump(data_C, f)
    print('Data C simulated and saved...')
    print('All data simulated and saved !')
    

if __name__ == '__main__':
    main()