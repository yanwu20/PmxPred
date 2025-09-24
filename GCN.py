import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn

# GCN without sequence alignment with 

class GCN_pmx(torch.nn.Module):
    def __init__(self, hidden_channels, node_feature_dim):
        super(GCN_pmx, self).__init__()
        torch.manual_seed(12345)
        self.hidden_size = [512, 256, 128]
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.lin1 = Linear(12*hidden_channels,64)
        self.lin2 = Linear(64, 1)
        self.compress2 = nn.Sequential(
            Linear(hidden_channels*2, 64),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch,miss_index):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x_temp = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu() + x_temp

        padding = torch.zeros(1, x.shape[1]).cuda()
        for i, v in enumerate(miss_index):
            batch_index = batch[v]
            true_index = v - sum([1 if b < batch_index else 0 for b in batch]) + batch_index * 12
            x = torch.cat([x[:true_index], padding, x[true_index:]], 0)
        x = x.view(max(batch) + 1, -1)

        x = self.lin1(x)
        x = self.lin2(x)

        return torch.sigmoid(x)

# simple GCN

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, node_feature_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        # self.bn1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        # self.bn2 = nn.LayerNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = self.lin2(x)

        return torch.sigmoid(x)

# GCN with sequence alignment

class GCN_ali(torch.nn.Module):
    def __init__(self, hidden_channels, node_feature_dim):
        super(GCN_ali, self).__init__()
        torch.manual_seed(12345)
        self.hidden_size = [512, 256, 128]
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.lin1 = Linear(12*hidden_channels,64)
        self.lin2 = Linear(64, 1)
        self.compress2 = nn.Sequential(
            Linear(hidden_channels*2, 64),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x_temp = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu() + x_temp

        x = x.view(max(batch) + 1, -1)

        x = self.lin1(x)
        x = self.lin2(x)

        return torch.sigmoid(x)

# GCN with pooling

class GCN_max(torch.nn.Module):
    def __init__(self, hidden_channels, node_feature_dim):
        super(GCN_max, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        # self.bn1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        # self.bn2 = nn.LayerNorm(hidden_channels)
        self.lin1 = Linear(2*hidden_channels+node_feature_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        concat_data = global_max_pool(x, batch)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        concat_data = torch.cat((concat_data, global_max_pool(x, batch)), 1)
        # x_temp = x
        #
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        # x = x.relu() + x_temp
        x = global_mean_pool(x, batch)
        concat_data = torch.cat((concat_data, x), 1)
        x = F.dropout(concat_data, p=0.4, training=self.training)
        x = self.lin1(x)
        # x = x.relu()
        x = self.lin2(x)

        return torch.sigmoid(x)