from param_parser import parameter_parser_train
from models.cmlp_trainer import CMLP_Trainer


def main():
    args = parameter_parser_train()
    cmlp_trainer = CMLP_Trainer(args)
    cmlp_trainer.train_model_ista()
    cmlp_trainer.save_model_and_loss()
    cmlp_trainer.reset_model()


if __name__ == '__main__':
    main()
