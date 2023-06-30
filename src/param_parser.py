"""Getting params from the command line."""

from argparse import ArgumentParser

def parameter_parser_train():
    """
    A method to parse up command line parameters.
    """
    parser = ArgumentParser(description="Train CMLP")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../data",
        help="Folder with simulated VAR data."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="../configs/train_cmlp.yaml",
        help="Path to cmlp training yaml."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./saved_models",
        help="Path to cmlp training yaml."
    )
    parser.add_argument(
        "--data_catagory",
        type=str,
        default="retail_latent",
        help="Data catagory."
    )
    
    return parser.parse_args()

def parameter_parser_simulation():
    """
    A method to parse up command line parameters.
    """
    parser = ArgumentParser(description="Simulate VAR data")

    parser.add_argument(
        "--save_path",
        type=str,
        default="../data",
        help="Folder to save simulated VAR data."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="../configs/simulation.yaml",
        help="Path to data simulation yaml."
    )
    
    return parser.parse_args()
