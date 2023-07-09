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
    
    endogenous = VAR.simulate_var_endogenous(cfg['endogenous'])
    latent = VAR.simulate_var_latent(cfg['latent'])
    retail_latent = VAR.simulate_var_retail_latent(cfg['retail_latent'])
    
    with open(f"{args.save_dir}/endogenous.pickle", 'wb') as f:
        pickle.dump(endogenous, f)
    print('Endogenous data simulated and saved...')

    with open(f"{args.save_dir}/latent.pickle", 'wb') as f:
        pickle.dump(latent, f)
    print('Latent VAR data simulated and saved...')

    with open(f"{args.save_dir}/retail_latent.pickle", 'wb') as f:
        pickle.dump(retail_latent, f)
    print('Retail latent VAR data simulated and saved...')
    
    print('All data simulated and saved !')
    

if __name__ == '__main__':
    main()