import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='SNN_TL parameters')

    parser.add_argument('--name', default='SNN_TL', type=str, help='the model name')
    parser.add_argument('--project_path', default='./', type=str, help='the project path')

    # data path
    parser.add_argument('--ROOT_PATH', default='PACS', type=str, help='the root path of data')
    parser.add_argument('-S', '--SOURCE_NAME', default='cartoon', type=str, help='the source data')
    parser.add_argument('-T', '--TARGET_NAME', default='sketch', type=str, help='the target data')

    # general
    parser.add_argument('--base_batch_size', default=30, type=int, help='the batch size of each GPU')
    parser.add_argument('--num_classes', default=7, type=int)

    # train
    parser.add_argument('--num_epochs', default=200, type=int, help='the number of epochs')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='the learning rate of train')

    # model
    parser.add_argument('--model_structs', default='alexnet', type=str, help='the network structs')
    parser.add_argument('--thresh', default=0.5, type=float, help='neuronal threshold')
    parser.add_argument('--lens', default=0.5, type=float, help='hyper-parameters of approximate function')
    parser.add_argument('--decay', default=0.5, type=float, help='decay constants')
    parser.add_argument('--time_window', default=12, type=int, help='the time step of SNN')

    # transfer loss
    parser.add_argument('-F', '--transfer_function', default="linear_CKA",
                        type=str, help='the loss function of transfer')
    parser.add_argument('-C', '--clf_function', default="Cross",
                        type=str, help='the loss function of clf')
    parser.add_argument('-R', '--transfer_ratio', default=0.1, type=float, help='the ratio of domain loss')
    parser.add_argument('-L', '--transfer_level', default=2, choices=[1, 2, 3, 4, 5], type=int,
                        help='the level of domain loss in the structure')

    return parser

