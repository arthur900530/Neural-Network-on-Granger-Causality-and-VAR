from param_parser import parameter_parser_train
from models.cmlp_trainer import CMLP_Container


def main():
    args = parameter_parser_train()
    cmlp_container = CMLP_Container(args=args)
    cmlp_container.train_model_ista()
    cmlp_container.save_model_and_loss()
    result = cmlp_container.evaluate()
    print(result)
    cmlp_container.reset_model()


if __name__ == '__main__':
    main()
