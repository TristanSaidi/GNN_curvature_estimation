import argparse
import torch
import sys
from src.trainers.gnn_trainer import GCNTrainer

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Train a GNN to predict scalar curvature')

    parser.add_argument(
        '--data_dir',
        default=
        'data',
        help='path to data files'
    )
    parser.add_argument(
        '--save_dir',
        default=
        'outputs',
        help='path to saved model files'
    )
    parser.add_argument(
        '--exp_name',
        default='gcn',
        help='name of experiment'
    )
    parser.add_argument(
        '--hidden_channels',
        default=64,
        type=int,
        help='number of times to repeat experiment'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='number of times to repeat experiment'
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='mini-batch size used to train model'
    )
    parser.add_argument(
        '--subgraph_k',
        default=2,
        type=int,
        help='number of hops to use in constructing subgraphs'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='learning rate for optimizer'
    )
    parser.add_argument(
        '--split',
        default=0.8,
        type=float,
        help='proportion of data to use for training'
    )
    parser.add_argument(
        '--seed',
        default=11202022,
        type=int,
        help='random seed to be used in numpy and torch'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='cpu or gpu ID to use'
    )

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs.pop('seed'))

    epochs = configs.pop('epochs')
    trainer = GCNTrainer(**configs)
    trainer.train(epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())