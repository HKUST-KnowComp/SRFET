from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, hidden_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(dropout)

        if n_layers == 1:
            self.linear_map = nn.Linear(input_dim, output_dim, bias=False)
        elif n_layers == 2:
            mlp_hidden_dim = input_dim // 2 if hidden_dim is None else hidden_dim
            self.linear_map1 = nn.Linear(input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            # self.linear_map2 = nn.Linear(mlp_hidden_dim, type_embed_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, output_dim)
        elif n_layers == 3:
            mlp_hidden_dim = input_dim // 2 if hidden_dim is None else hidden_dim
            self.linear_map1 = nn.Linear(input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, output_dim)
        elif n_layers == 4:
            mlp_hidden_dim = input_dim // 2 if hidden_dim is None else hidden_dim
            self.linear_map1 = nn.Linear(input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin3_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map4 = nn.Linear(mlp_hidden_dim, output_dim)

    def forward(self, input_vecs):
        output_vecs = None
        if self.n_layers == 1:
            output_vecs = self.linear_map(input_vecs)
        elif self.n_layers == 2:
            l1_output = self.linear_map1(input_vecs)
            l1_output = self.lin1_bn(F.relu(l1_output))
            output_vecs = self.linear_map2(self.dropout_layer(l1_output))
        elif self.n_layers == 3:
            l1_output = self.linear_map1(input_vecs)
            l1_output = self.lin1_bn(F.relu(l1_output))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            output_vecs = self.linear_map3(self.dropout_layer(l2_output))
        elif self.n_layers == 4:
            l1_output = self.linear_map1(input_vecs)
            l1_output = self.lin1_bn(F.relu(l1_output))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            l3_output = self.linear_map3(self.dropout_layer(l2_output))
            l3_output = self.lin3_bn(F.relu(l3_output))
            output_vecs = self.linear_map4(self.dropout_layer(l3_output))
        return output_vecs
