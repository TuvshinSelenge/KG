import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch.nn import Module, Linear
import torch.nn.functional as F

class RGCN(Module):
    def __init__(self,in_c,hid_c,n_rel):
        super().__init__()
        self.conv1 = RGCNConv(in_c,hid_c,n_rel)
        self.conv2 = RGCNConv(hid_c,hid_c,n_rel)
        self.lin   = Linear(hid_c,hid_c)
    def forward(self,x,ei,et):
        x = F.relu(self.conv1(x,ei,et))
        x = self.conv2(x,ei,et)
        return self.lin(x)
