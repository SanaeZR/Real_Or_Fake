import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool as gmp

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1, dropout=0.5):
        super().__init__()
      
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.lin_news = nn.Linear(in_channels, hidden_channels)
        
        
       
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin0 = nn.Linear(hidden_channels, hidden_channels)
        
        
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.lin1 = nn.Linear(2 * hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        # Layer 1 
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3 
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        
        h = gmp(h, batch)
        h = self.lin0(h)
        h = F.relu(h)

        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin_news(news).relu()

        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)
