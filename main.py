import argparse
import torch
import sys
from src.trainers.gnn_trainer import GNNTrainer

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Train a GNN to predict scalar curvature')

    parser.add_argument(
        '--data_dir',
        default='data',
        help='path to data files'
    )
    parser.add_argument(
        '--save_dir',
        default='outputs',
        help='path to saved model files'
    )
    parser.add_argument(
        '--exp_name',
        default='gcn',
        help='name of experiment'
    )
    parser.add_argument(
        '--architecture',
        default='gat',
        choices=['gcn', 'gat'],
        help='type of architecture to use'
    )
    parser.add_argument(
        '--hidden_channels',
        default=256,
        type=int,
        help='number of hidden channels in model'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='number of times to repeat experiment'
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='mini-batch size used to train model'
    )
    parser.add_argument(
        '--subgraph_k',
        default=1,
        type=int,
        help='number of hops to use in constructing subgraphs'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='learning rate for optimizer'
    )
    parser.add_argument(
        '--degree_features',
        default=False,
        type=bool,
        help='whether to use degree features in GAT model'
    )
    parser.add_argument(
        '--num_layers',
        default=5,
        type=int,
        help='number of layers in model'
    )
    parser.add_argument(
        '--dropout',
        default=0.1,
        type=float,
        help='dropout rate to use in model'
    )
    parser.add_argument(
        '--split',
        default=0.8,
        type=float,
        help='proportion of data to use for training'
    )
    parser.add_argument(
        '--manifold_split',
        default=False,
        type=bool,
        help='whether to split data by manifold'
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
    trainer = GNNTrainer(**configs)
    trainer.train(epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())