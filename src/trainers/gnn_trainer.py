import torch
from src.models.gnn import GCNRegressor, GATRegressor
from src.datasets.dataset import ManifoldGraphDataset
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

architectures = {
    'gcn': GCNRegressor,
    'gat': GATRegressor
}

class GNNTrainer(object):
    def __init__(self, 
                 data_dir, 
                 save_dir,
                 exp_name, 
                 subgraph_k,
                 architecture,
                 hidden_channels, # model hyperparameters
                 degree_features,
                 batch_size, 
                 learning_rate,
                 num_layers,
                 dropout,
                 split, 
                 manifold_split,
                 device):
        super(GNNTrainer, self).__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'nn'), exist_ok=True)
        self.subgraph_k = subgraph_k
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.architecture = architecture
        self.degree_features = degree_features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # if random split
        self.split = split # train/val split pctg
        # if manifold split
        self.manifold_split = manifold_split

        self.device = device
        # setup logging
        self.train_writer = SummaryWriter(os.path.join(self.save_path, 'logs_train'))
        self.val_writer = SummaryWriter(os.path.join(self.save_path, 'logs_val'))
        # load data and 
        if self.manifold_split:
            self.load_data_manifoldsplit()
        else:
            self.load_data_randsplit()
        # initialize model
        self.initialize_model()

    def load_data_manifoldsplit(self):
        # load data and split into train and val by manifold
        data_dir_train = os.path.join(self.data_dir, 'train')
        data_dir_val = os.path.join(self.data_dir, 'val')
        train_list = os.listdir(data_dir_train)
        val_list = os.listdir(data_dir_val)
        train_data = []
        val_data = []
        for file in train_list:
            print(f'Loading file {file} for training...')
            full_graph = torch.load(os.path.join(data_dir_train, file))
            train_data.append(full_graph)
        for file in val_list:
            print(f'Loading file {file} for validation...')
            full_graph = torch.load(os.path.join(data_dir_val, file))
            val_data.append(full_graph)
        train_dataset = ManifoldGraphDataset(train_data, self.subgraph_k, degree_features=self.degree_features)
        val_dataset = ManifoldGraphDataset(val_data, self.subgraph_k, degree_features=self.degree_features)
        self.num_node_features = train_dataset.num_node_features
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def load_data_randsplit(self):
        # load data and split into train and val randomly (pulling from all manifolds)
        file_list = os.listdir(self.data_dir)
        data = []
        for file in file_list:
            full_graph = torch.load(os.path.join(self.data_dir, file))['graph']
            data.append(full_graph)
        dataset = ManifoldGraphDataset(data, self.subgraph_k, self.degree_features)
        self.num_node_features = dataset.num_node_features
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*self.split), len(dataset) - int(len(dataset)*self.split)])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

    def initialize_model(self):
        in_channels = 11 if self.degree_features else self.num_node_features # degree can be {0,10}, thus 11 possible values for one-hot encoding
        hidden_channels = self.hidden_channels
        self.model = architectures[self.architecture](
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            num_layers = self.num_layers,
            dropout = self.dropout
        ).to(self.device)
        print(f'Number of parameters in model: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, epochs):
        val_loss_min = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.eval()
            # log training and validation loss
            self.train_writer.add_scalar('Loss', train_loss, epoch)
            self.val_writer.add_scalar('Loss', val_loss, epoch)
            if val_loss < val_loss_min:
                print(f'Epoch {epoch}: Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.save(self.model.state_dict())
                val_loss_min = val_loss
            else:
                print(f'Epoch {epoch}: Validation loss: {val_loss}')

    def save(self, state_dict):
        torch.save(state_dict, os.path.join(self.save_path, 'nn/best_val.pt'))

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x, edge_index, edge_attrs, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
            edge_weights = torch.exp(-1*edge_attrs)
            out = self.model(x=x, edge_index=edge_index, edge_weight=edge_weights, batch=batch)
            loss = F.mse_loss(out.squeeze(1), data.y.float())
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return running_loss / len(self.train_loader)
    
    def eval(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                x, edge_index, edge_attrs, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
                edge_weights = torch.exp(-1*edge_attrs)
                out = self.model(x=x, edge_index=edge_index, edge_weight=edge_weights, batch=batch)
                loss = F.mse_loss(out.squeeze(1), data.y.float())
                running_loss += loss.item()
        return running_loss / len(self.val_loader)