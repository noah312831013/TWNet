import torch
import torch.nn as nn

from model.base_model import base_model
from model.modules.horizon_net_feature_extractor import HorizonNetFeatureExtractor
from model.modules.transformer import Transformer

class TWNet(base_model):
    def __init__(self, dropout=0.0, win_size=8, depth=6):
        super().__init__()

        self.name = 'TWNet'
        self.patch_num = 256
        self.patch_dim = 1024
        self.dropout_d = dropout
        self.feature_extractor = HorizonNetFeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        transformer_dim = self.patch_dim
        transformer_layers = depth
        transformer_heads = 8
        transformer_head_dim = transformer_dim // transformer_heads
        transformer_ff_dim = 1024
        self.transformer = Transformer(dim=transformer_dim, 
                                       depth=transformer_layers,
                                       heads=transformer_heads, 
                                       dim_head=transformer_head_dim,
                                       mlp_dim=transformer_ff_dim, win_size=win_size,
                                       dropout=self.dropout_d, patch_num=self.patch_num)
        
        self.linear = nn.Linear(in_features=self.patch_dim, out_features=1)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        # output = torch.clamp(x.view(-1,self.patch_num),0,1)
        return x
    