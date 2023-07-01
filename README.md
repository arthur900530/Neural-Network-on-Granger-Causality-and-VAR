# Neural-Network-on-Granger-Causality-and-VAR
## Environment Requirements
```
torch == 2.0.1
seaborn == 0.12.2
numpy == 1.24.3
tqdm == 4.65.0
```
## Run the Code
### Stimulation:
```
python var_simulation.py
```
```
--save_dir         STR      Directory to save simulated VAR data.     Default is "../data".
--yaml_path        STR      Directory to save simulated VAR data.     Default is "../data".
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
