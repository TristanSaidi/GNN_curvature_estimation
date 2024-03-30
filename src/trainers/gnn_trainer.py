import torch
from src.models.gnn import GCNRegressor, GATRegressor, GCNClassifier
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
    'gcn_clf': GCNClassifier,
    'gat': GATRegressor
}

class GNNTrainer(object):
    def __init__(self, 
                 data_dir, 
                 save_dir,
                 exp_name, 
                 task,
                 subgraph_k,
                 scale_features,
                 edge_attrs,
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
        self.task = task
        assert self.task in ['regression', 'classification']

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'nn'), exist_ok=True)
        self.subgraph_k = subgraph_k
        self.scale_features = scale_features
        print(f'Scaling features: {self.scale_features}')
        # assert not ('nbr_distances' not in self.data_dir and self.scale_features), 'Cannot scale features if not using nbr_distances'
        self.edge_attrs = edge_attrs

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.architecture = 'gcn_clf' if self.task == 'classification' else architecture
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
        train_data = {}
        val_data = {}
        for file in train_list:
            print(f'Loading file {file} for training...')
            full_graph = torch.load(os.path.join(data_dir_train, file))
            train_data[file] = full_graph
        for file in val_list:
            print(f'Loading file {file} for validation...')
            full_graph = torch.load(os.path.join(data_dir_val, file))
            val_data[file] = full_graph
        train_dataset = ManifoldGraphDataset(self.task, train_data, self.subgraph_k, degree_features=self.degree_features, subsample_pctg=0.5, scale_features=self.scale_features, edge_attrs=self.edge_attrs)
        val_dataset = ManifoldGraphDataset(self.task, val_data, self.subgraph_k, degree_features=self.degree_features, subsample_pctg=0.05, scale_features=self.scale_features, edge_attrs=self.edge_attrs)

        self.num_node_features = train_dataset.num_node_features
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def load_data_randsplit(self):
        # load data and split into train and val randomly (pulling from all manifolds)
        file_list = os.listdir(self.data_dir)
        data = {}
        for file in file_list:
            full_graph = torch.load(os.path.join(self.data_dir, file))
            data[file] = full_graph
        dataset = ManifoldGraphDataset(self.task, data, self.subgraph_k, self.degree_features, scale_features=self.scale_features, edge_attrs=self.edge_attrs)

        self.num_node_features = dataset.num_node_features
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*self.split), len(dataset) - int(len(dataset)*self.split)])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

    def initialize_model(self):
        in_channels = self.num_node_features
        hidden_channels = self.hidden_channels
        self.model = architectures[self.architecture](
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            num_layers = self.num_layers,
            dropout = self.dropout
        ).to(self.device)
        print(f'Number of parameters in model: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)
    
    def train(self, epochs):
        val_loss_min = float('inf')
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                train_loss = self.train_epoch()
                val_loss, acc = self.eval()
                # log training and validation loss
                self.train_writer.add_scalar('Loss', train_loss, epoch)
                self.val_writer.add_scalar('Loss', val_loss, epoch)
                if self.task == 'classification': self.val_writer.add_scalar('Accuracy', acc, epoch)
                if val_loss < val_loss_min:
                    self.save()
                    val_loss_min = val_loss
                str = f'Min val loss: {val_loss_min:.2f}, Val acc: {acc:.2f}, Train loss: {train_loss:.2f}' if self.task == 'classification' else f'Min val loss: {val_loss_min:.2f}, Train loss: {train_loss:.2f}'
                pbar.set_postfix_str(str)
                pbar.update(1)

    def save(self):
        state_dict = self.model.state_dict()
        save_dict = {
            "model": state_dict,
        }
        torch.save(save_dict, os.path.join(self.save_path, 'nn/best_val.pt'))

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x, edge_index, edge_attrs, batch, y = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch, data.y.float()
            # forward pass
            y_hat = self.model(x=x, edge_index=edge_index, edge_weight=edge_attrs, batch=batch)
            # loss
            if self.task == 'classification':
                # in this case, y contains class indices
                loss = F.cross_entropy(y_hat, y.long())
            elif self.task == 'regression':
                # in this case, y contains scalar values
                loss = F.mse_loss(y_hat.squeeze(1), y, reduction='mean')
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return running_loss / len(self.train_loader)
    
    def eval(self):
        self.model.eval()
        running_loss = 0.0
        
        # accuracy logging for classification task
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                x, edge_index, edge_attrs, batch, y = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch, data.y.float()
                # scale input and output
                # forward pass
                y_hat = self.model(x=x, edge_index=edge_index, edge_weight=edge_attrs, batch=batch)

                if self.task == 'classification':
                    # in this case, y contains class indices
                    loss = F.cross_entropy(y_hat, y.long())
                    # compute accuracy
                    _, preds = torch.max(y_hat, 1)
                    correct = (preds == y).sum().item()
                    total = y.size(0)
                    # update correct and total
                    correct += correct
                    total += total
                elif self.task == 'regression':
                    # in this case, y contains scalar values
                    loss = F.mse_loss(y_hat.squeeze(1), y, reduction='mean')
                running_loss += loss.item()
        accuracy = correct / total if self.task == 'classification' else None
        return running_loss / len(self.val_loader), accuracy