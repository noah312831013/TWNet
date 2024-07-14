from model.modules.transformer_modules import *


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, win_size, dim_head, mlp_dim,
                 dropout=0., patch_num=None, ape=None, rpe=None, rpe_pos=1):
        super().__init__()

        self.absolute_pos_embed = None if patch_num is None or ape is None else AbsolutePosition(dim, dropout,
                                                                                                 patch_num, ape)
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, patch_num=patch_num,
                                       rpe=rpe, rpe_pos=rpe_pos)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        if self.absolute_pos_embed is not None:
            x = self.absolute_pos_embed(x)
        x = self.pos_dropout(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
