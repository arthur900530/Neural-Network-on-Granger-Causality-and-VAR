# Neural-Network-on-Granger-Causality-and-VAR

This repo contains implementation of Neural-Network-on-Granger-Causality-and-VAR, the data we simulated can be divided into three catagories: endogenous, exdogenogenous, and retail data. The code for data simulation is at *src/simulation*, code for deep learning models is at *src/models*. Our implementation is based on <cite>Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. B. (2021). Neural granger causality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8), 4267-4279.</cite>

## Environment Requirements
```
torch == 2.0.1
seaborn == 0.12.2
numpy == 1.24.3
tqdm == 4.65.0
```
## Run the Code
### Simulation:
```
python var_simulation.py
```
```
--save_dir         STR      Directory to save simulated VAR data.     Default is "../data".
--yaml_path        STR      Path to data simulation yaml.             Default is "../configs/simulation.yaml".
```
### Train cMLP:
```
python train_cmlp.py
```
```
--data_dir         STR      Directory with simulated VAR data.       Default is "../data".
--yaml_path        STR      Path to cmlp training yaml.              Default is "../configs/train_cmlp.yaml"
--model_save_dir   STR      Directory to save trained model.         Default is "./saved_models".
--catagory         STR      Data catagory.                           Default is "retail_latent"

```
## YAML Parameters
### Train cMLP
```
- model:
  - device: where to run the model.                                  Default is 'cuda'.
  - lag: the time lag of data.                                       Default is 1.
  - hidden: the number of cMLP model's hidden nodes.                 Default is 100.
  - lam: parameter for nonsmooth regularization.                     Default is 0.0022.
  - lam_ridge: parameter for ridge regularization on output layer.   Default is 0.015.
  - lr: the learning rate of training process.                       Default is 0.05.
  - penalty: the optimizztion penalty type.                          Default is 'H'.
  - max_iter: the number of training epochs.                         Default is 50000.
  - check_every: report the training information every n epochs.     Default is 100.
  - activation: the activation of cMLP model.                        Default is 'tanh'.
```
