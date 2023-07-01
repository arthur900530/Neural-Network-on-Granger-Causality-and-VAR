# Neural-Network-on-Granger-Causality-and-VAR
## Environment Requirements
```
torch == 2.0.1
seaborn == 0.12.2
numpy == 1.24.3
tqdm == 4.65.0
```

## YAML Parameters

- "ID": The ID of the target document, e.g. "12CR000002-V005".

- "QUERY": List consists of one or many Process ID.

- "TOPN": Determines the returned number _n_ of the processes with highest similarity value.

- "DOCUMENT_GRAPH_WEIGHT": Weight of the **Document Graph Similarity**.

- "FLOW_CHART_WEIGHT": Weight of the **Flow Chart Similarity**.

- "TEXT_WEIGHT": Weight of the **Text Similarity**.

- "SYSTEM_WEIGHT": Weight of the **System Similarity**.
  

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