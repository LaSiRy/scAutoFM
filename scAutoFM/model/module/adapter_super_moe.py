import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np
from lib.custom_layer import CustomLinear

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper_Moe(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 hypernet_hidden_size=64,
                 arch_dim=12,
                 max_experts=2,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper_Moe, self).__init__()
    
        self.embed_dims = embed_dims
        self.hypernet_hidden_size = hypernet_hidden_size
        self.max_experts = max_experts

        # Follow visual prompt
        # self.super_reductuion_dim = int(self.embed_dims/8)

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims

        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.adapter_router = torch.nn.Sequential(
                torch.nn.Linear(arch_dim, self.hypernet_hidden_size),
                torch.nn.ReLU(),
                CustomLinear(self.hypernet_hidden_size, self.max_experts * self.super_reductuion_dim)
            )
            self.ln1 = CustomLinear(self.embed_dims, self.super_reductuion_dim)
            # self.ln1_2 = CustomLinear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = CustomLinear(self.super_reductuion_dim, self.embed_dims)
            # self.ln2_2 = CustomLinear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)


    def set_sample_config(self, arch_embed, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            self.arch_embed = arch_embed.to(next(self.parameters()).device)
            self.ln1.set_sample_config(self.embed_dims, self.sample_embed_dim)
            # self.ln1_2.set_sample_config(self.embed_dims, self.sample_embed_dim)
            self.ln2.set_sample_config(self.sample_embed_dim, self.embed_dims)
            # self.ln2_2.set_sample_config(self.sample_embed_dim, self.embed_dims)

            self.adapter_router[-1].set_sample_config(self.hypernet_hidden_size, self.max_experts*sample_embed_dim)


    def forward(self, x, identity=None):
        if self.identity:
            # import pdb;pdb.set_trace()
            return x
            # return x + 0*self.sampled_weight_0 + 0*self.sampled_bias_0 + 0*self.sampled_weight_1 + 0*self.sampled_bias_1
        # import pdb;pdb.set_trace()
        adapter_expert_out = self.adapter_router(self.arch_embed)
        adapter_expert_out = adapter_expert_out.view(-1, self.max_experts)
        adapter_expert_out = torch.nn.Softmax(dim=-1)(adapter_expert_out)
        ln1_weights = self.ln1.samples["weight"] * adapter_expert_out[:, 0].view(-1,1)
                            #  + self.ln1_2.samples["weight"] * adapter1_expert_out[:, 1].view(-1,1)
        # ln1_bias = (self.ln1_1.samples["bias"] * adapter1_expert_out[:, 0]
        #             + self.ln1_2.samples["bias"] * adapter1_expert_out[:, 1])
        ln2_weights = self.ln2.samples["weight"] * adapter_expert_out[:, 0].view(1,-1)
                        # + self.ln2_2.samples["weight"] * adapter2_expert_out[:, 1].view(1,-1)
        # ln2_bias = (self.ln2_1.samples["bias"] * adapter2_expert_out[:, 0]
        #                 + self.ln2_2.samples["bias"] * adapter2_expert_out[:, 1])
    
        out = F.linear(x, ln1_weights)
        out = self.activate(out)
        out = self.dropout(out)
        out = F.linear(out, ln2_weights)
        if identity is None:
            identity = x
        return identity + out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return self.ln1.samples["weight"].numel() + self.ln2.samples["weight"].numel() \
                 + self.adapter_router[-1].samples["weight"].numel() 

            # return self.ln1_1.samples["weight"].numel() + self.ln1_2.samples["weight"].numel() \
            #      + self.ln2_1.samples["weight"].numel() + self.ln2_2.samples["weight"].numel() \
            #      + self.adapter_router_1[-1].samples["weight"].numel() + self.adapter_router_2[-1].samples["weight"].numel()
