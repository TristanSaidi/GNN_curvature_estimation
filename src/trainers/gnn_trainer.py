import torch
from src.models.gnn import GCNRegressor
from src.datasets.dataset import ManifoldGraphDataset
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import os

class GCNTrainer(object):
    def __init__(self, 
                 data_dir, 
                 save_dir,
                 exp_name, 
                 subgraph_k,
                 hidden_channels, # model hyperparameters
                 batch_size, 
                 learning_rate,
                 split, 
                 device):
        super(GCNTrainer, self).__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'nn'), exist_ok=True)
        self.subgraph_k = subgraph_k
        self.hidden_channels = hidden_channels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.split = split
        self.device = device

        # setup logging
        self.train_writer = SummaryWriter(os.path.join(self.save_path, 'logs_train'))
        self.val_writer = SummaryWriter(os.path.join(self.save_path, 'logs_val'))
        # load data and 
        self.load_data()
        # initialize model
        self.initialize_model()

    def load_data(self):
        dataset = ManifoldGraphDataset(self.data_dir, self.subgraph_k)
        self.num_node_features = dataset.num_node_features
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*self.split), len(dataset) - int(len(dataset)*self.split)])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

    def initialize_model(self):
        self.model = GCNRegressor(num_node_features=self.num_node_features, hidden_channels=self.hidden_channels).to(self.device)
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
            out = self.model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            loss = F.mse_loss(out, data.y.unsqueeze(1).float())
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
                out = self.model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
                loss = F.mse_loss(out, data.y.unsqueeze(1).float())
                running_loss += loss.item()
        return running_loss / len(self.val_loader)