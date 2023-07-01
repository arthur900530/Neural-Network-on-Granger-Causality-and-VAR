import yaml
from simulation import VAR
import pickle
from param_parser import parameter_parser_simulation

def main():
    args = parameter_parser_simulation()

    with open(args.yaml_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    data_A = VAR.simulate_var_endogenous(cfg['endogenous'])
    data_B = VAR.simulate_var_latent(cfg['latent'])
    data_C = VAR.simulate_var_retail_latent(cfg['retail_latent'])

    with open(f"{args.save_dir}/{cfg['data_catagory']['A']}.pickle", 'wb') as f:
        pickle.dump(data_A, f)
    print('Endogenous data simulated and saved...')

    with open(f"{args.save_dir}/{cfg['data_catagory']['B']}.pickle", 'wb') as f:
        pickle.dump(data_B, f)
    print('Latent VAR data simulated and saved...')

    with open(f"{args.save_dir}/{cfg['data_catagory']['C']}.pickle", 'wb') as f:
        pickle.dump(data_C, f)
    print('Retail latent VAR data simulated and saved...')
    
    print('All data simulated and saved !')
    

if __name__ == '__main__':
    main()