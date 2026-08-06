import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
import math
import copy
from lib.custom_layer import CustomLinear
from model.module.adapter_super_moe import AdapterSuper_Moe
import numpy as np
from timm.models.layers import trunc_normal_, lecun_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.02, proj_drop=0., LoRA_dim=1024, prefix_dim=1024, hypernet_hidden_size=64, arch_dim=12, max_experts=1, drop_rate_LoRA=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.hypernet_hidden_size = hypernet_hidden_size
        self.max_experts = max_experts
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.LoRA_dim = LoRA_dim
        self.prefix_dim = prefix_dim

        # 用于模块专家路由选择
        self.LoRA_router_1 = torch.nn.Sequential(
            torch.nn.Linear(arch_dim, self.hypernet_hidden_size),
            torch.nn.ReLU(),
            CustomLinear(self.hypernet_hidden_size, self.max_experts * self.LoRA_dim)
        )

        self.LoRA_router_2 = torch.nn.Sequential(
            torch.nn.Linear(arch_dim, self.hypernet_hidden_size),
            torch.nn.ReLU(),
            CustomLinear(self.hypernet_hidden_size, self.max_experts * self.LoRA_dim)
        )
        
        if LoRA_dim > 0:
            # DoRA
            self.query_LoRA_m = nn.Parameter(torch.ones(dim)) 
            self.value_LoRA_m = nn.Parameter(torch.ones(dim))           

            # DoRA
            self.LoRA_a_expert_q = CustomLinear(self.dim, self.LoRA_dim)
            self.LoRA_a_expert_v = CustomLinear(self.dim, self.LoRA_dim)
            self.LoRA_b_expert_q = CustomLinear(self.LoRA_dim, self.dim)
            self.LoRA_b_expert_v = CustomLinear(self.LoRA_dim, self.dim)

        
        if prefix_dim > 0:
            self.prefix_tokens_key = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            self.prefix_tokens_value = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            self.prefix_token_gate = nn.Parameter(torch.ones(dim, prefix_dim))
            
            self.prefix_layer_gate = nn.Parameter(torch.tensor(1.0))
            nn.init.xavier_uniform_(self.prefix_token_gate)
            
            nn.init.normal_(self.prefix_tokens_key, mean=0.0, std=0.02)
            nn.init.normal_(self.prefix_tokens_value, mean=0.0, std=0.02)


        self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
        drop_rate_prefix = drop_rate_LoRA
        self.prefix_drop = nn.Dropout(p=drop_rate_prefix)

    def set_sample_config(self, arch_embed, sample_DoRA_dim,sample_prefix_dim):
        self.sample_DoRA_dim = sample_DoRA_dim
        self.LoRA_identity = False
        if self.sample_DoRA_dim == 0:
            self.LoRA_identity = True 
        else:
            self.arch_embed = arch_embed.to(next(self.parameters()).device)
            self.LoRA_a_expert_q.set_sample_config(self.dim, self.sample_DoRA_dim)
            self.LoRA_a_expert_v.set_sample_config(self.dim, self.sample_DoRA_dim)
            self.LoRA_b_expert_q.set_sample_config(self.sample_DoRA_dim, self.dim)
            self.LoRA_b_expert_v.set_sample_config(self.sample_DoRA_dim, self.dim)

            self.LoRA_router_2[-1].set_sample_config(self.hypernet_hidden_size, self.max_experts*self.sample_DoRA_dim)
            self.LoRA_router_1[-1].set_sample_config(self.hypernet_hidden_size, self.max_experts*self.sample_DoRA_dim)


        self.sample_prefix_dim = sample_prefix_dim
        self.prefix_identity = False
        if self.sample_prefix_dim == 0:
            self.prefix_identity = True 
        else:
            self.prefix_weight_key = self.prefix_tokens_key[:,:self.sample_prefix_dim,:]
            self.prefix_weight_value = self.prefix_tokens_value[:,:self.sample_prefix_dim,:]
            self.prefix_token_gate_weight = self.prefix_token_gate[:,:self.sample_prefix_dim]
        
    def calc_sampled_param_num(self):
        if self.sample_DoRA_dim == 0:
            return 0
        else:
            return self.LoRA_a_expert_q.samples["weight"].numel() + self.LoRA_a_expert_v.samples["weight"].numel() \
                 + self.LoRA_b_expert_q.samples["weight"].numel() + self.LoRA_b_expert_v.samples["weight"].numel() \
                 + self.LoRA_router_1[-1].samples["weight"].numel() + self.LoRA_router_2[-1].samples["weight"].numel()

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if self.LoRA_identity == False:
            
            lora_expert_out_q = self.LoRA_router_1(self.arch_embed)
            lora_expert_out_q = lora_expert_out_q.view(-1, self.max_experts)
            lora_expert_out_q = torch.nn.Softmax(dim=-1)(lora_expert_out_q)
            lora_expert_out_v = self.LoRA_router_2(self.arch_embed)
            lora_expert_out_v = lora_expert_out_v.view(-1, self.max_experts)
            lora_expert_out_v = torch.nn.Softmax(dim=-1)(lora_expert_out_v)
            # LoRA_a_weights = ( self.LoRA_a_expert_1.samples["weight"] * lora_expert_out_1[:, 0].view(-1,1)
            #                  + self.LoRA_a_expert_2.samples["weight"] * lora_expert_out_1[:, 1].view(-1,1))
            # LoRA_a_bias = (self.LoRA_a_expert_1.samples["bias"] * lora_expert_out_1[:, 0]
            #             + self.LoRA_a_expert_2.samples["bias"] * lora_expert_out_1[:, 1])
            # LoRA_b_weights = (self.LoRA_b_expert_1.samples["weight"] * lora_expert_out_2[:, 0].view(1,-1)
                            # + self.LoRA_b_expert_2.samples["weight"] * lora_expert_out_2[:, 1].view(1,-1))
            # LoRA_b_bias = (self.LoRA_b_expert_1.samples["bias"] * lora_expert_out_2[:, 0]
            #              + self.LoRA_b_expert_2.samples["bias"] * lora_expert_out_2[:, 1])

            lora_a_weights_q = self.LoRA_a_expert_q.samples["weight"] * lora_expert_out_q[:, 0].view(-1,1)
            lora_a_weights_v = self.LoRA_a_expert_v.samples["weight"] * lora_expert_out_v[:, 0].view(-1,1)
            lora_b_weights_q = self.LoRA_b_expert_q.samples["weight"] * lora_expert_out_q[:, 0].view(1,-1)
            lora_b_weights_v = self.LoRA_b_expert_v.samples["weight"] * lora_expert_out_v[:, 0].view(1,-1)

            q_w = self.query.weight + lora_b_weights_q @ lora_a_weights_q
            v_w = self.value.weight + lora_b_weights_q @ lora_a_weights_v
            q_scale = self.query_LoRA_m / (q_w.norm(p=2, dim=0) + 1e-6)
            v_scale = self.value_LoRA_m / (v_w.norm(p=2, dim=0) + 1e-6)

            q_prime = F.linear(self.LoRA_drop(x), q_w)
            v_prime = F.linear(self.LoRA_drop(x), v_w)

            q = (q_prime * q_scale).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = (v_prime * v_scale).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        else:
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.prefix_identity == False:
            h_gate = x.mean(dim=1)  # shape: [B, C]
            alpha = torch.sigmoid(h_gate @ self.prefix_token_gate_weight)  # [B, l]
            lambda_ = torch.sigmoid(self.prefix_layer_gate)     

            prefix_k = self.prefix_weight_key.expand(B, -1, -1)
            prefix_v = self.prefix_weight_value.expand(B, -1, -1)

            # Apply token-level gate α and layer gate λ
            gated_weight = lambda_ * alpha.unsqueeze(-1)            # [B, l, 1]
            prefix_k = prefix_k * gated_weight                      # [B, l, C]
            prefix_v = prefix_v * gated_weight 

            prefix_k = prefix_k.reshape(B, self.sample_prefix_dim, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            prefix_v = prefix_v.reshape(B, self.sample_prefix_dim, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            prefix_k = self.prefix_drop(prefix_k)
            prefix_v = self.prefix_drop(prefix_v)

            k = torch.cat((prefix_k, k), dim=2)
            v = torch.cat((prefix_v, v), dim=2)

            prefix_attention_mask = torch.ones((attention_mask.size(0), self.sample_prefix_dim), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -1e9
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
            attn += attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class BertOutput(nn.Module):
    def __init__(self, hidden_size, output_size, layer_norm_eps, adapter_dim, drop_rate_adapter, hypernet_hidden_size=64, arch_dim=12, max_experts=1, drop_rate=0.02):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.LayerNorm = nn.LayerNorm(output_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(drop_rate)

        self.serial_adapter = AdapterSuper_Moe(
            embed_dims=output_size,
            reduction_dims=adapter_dim,
            hypernet_hidden_size=hypernet_hidden_size, 
            arch_dim=arch_dim, max_experts=max_experts,
            drop_rate_adapter=drop_rate_adapter
        )
        self.parallel_adapter = AdapterSuper_Moe(
            embed_dims=output_size,
            reduction_dims=adapter_dim,
            hypernet_hidden_size=hypernet_hidden_size, 
            arch_dim=arch_dim, max_experts=max_experts,
            drop_rate_adapter=drop_rate_adapter
        )

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.serial_adapter(hidden_states, hidden_states)
        hidden_states = self.parallel_adapter(input_tensor, input_tensor) + hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, output_size, layer_norm_eps, drop_rate=0.02):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_rate)
        self.LayerNorm = nn.LayerNorm(output_size, eps=layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        output = self.activation(self.linear(hidden_states))
        return output

# 自定义 Block 模块
class Block(nn.Module):
    def __init__(self, config, dim, num_heads, qkv_bias=False, attn_drop=0.,
                 drop_path=0., LoRA_dim=1024, adapter_dim=1024, prefix_dim=1024, hypernet_hidden_size=64, arch_dim=12, max_experts=1, drop_rate_LoRA=0, drop_rate_adapter=0):
        super().__init__()
        self.LoRA_dim = LoRA_dim
        self.adapter_dim = adapter_dim
        self.prefix_dim = prefix_dim
        self.attn = Attention(config.hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              LoRA_dim=LoRA_dim, prefix_dim=prefix_dim, hypernet_hidden_size=hypernet_hidden_size, arch_dim=arch_dim, max_experts=max_experts, drop_rate_LoRA=drop_rate_LoRA)
        
        self.attn_output = BertSelfOutput(hidden_size=config.hidden_size, output_size=config.hidden_size, layer_norm_eps=config.layer_norm_eps)

        # DropPath for stochastic depth
        self.bertintermediate = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.intermediate_act_fn = nn.ReLU()

        self.output = BertOutput(hidden_size=config.intermediate_size, output_size=config.hidden_size, layer_norm_eps=config.layer_norm_eps, adapter_dim=adapter_dim, 
                    hypernet_hidden_size=hypernet_hidden_size, arch_dim=arch_dim, max_experts=max_experts, drop_rate_adapter=drop_rate_adapter)


    def set_sample_config(self, sample_DoRA_dim=None,sample_s_adapter_dim=None,sample_p_adapter_dim=None,sample_prefix_dim=None):

        def encode(sample_dim, max_dim, min_unit=128):
            if sample_dim is None or sample_dim == 0:
                return [0.0, 0.0, 0.0]
            return [
                sample_dim / max_dim,
                np.log1p(sample_dim) / np.log1p(max_dim),
                sample_dim / min_unit
            ]
        arch_list = []
        min_unit = 5
        arch_list += encode(sample_DoRA_dim, self.LoRA_dim, min_unit)
        arch_list += encode(sample_s_adapter_dim, self.adapter_dim, min_unit)
        arch_list += encode(sample_p_adapter_dim, self.adapter_dim, min_unit)
        arch_list += encode(sample_prefix_dim, self.prefix_dim, min_unit)
        
        arch_embed = torch.tensor(arch_list, dtype=torch.float32)

        self.attn.set_sample_config(arch_embed=arch_embed, sample_DoRA_dim=sample_DoRA_dim,sample_prefix_dim=sample_prefix_dim)
        self.sample_s_adapter_dim = sample_s_adapter_dim
        self.sample_p_adapter_dim = sample_p_adapter_dim
        self.output.parallel_adapter.set_sample_config(arch_embed=arch_embed, sample_embed_dim=self.sample_p_adapter_dim)
        self.output.serial_adapter.set_sample_config(arch_embed=arch_embed, sample_embed_dim=self.sample_s_adapter_dim)

    def calc_sampled_param_num(self):
        return 0

    def forward(self, x, attention_mask):
        attention_output = self.attn_output(self.attn(x, attention_mask), x)
        intermediate_output = self.intermediate_act_fn(self.bertintermediate(attention_output))
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class GeneFormer_MOE(nn.Module):
    def __init__(self, config, basemodel, num_classes, pool=False, weight_init='',LoRA_dim=1024,adapter_dim=1024,prefix_dim=1024,drop_rate_LoRA=0.1,drop_rate_prompt=0,drop_rate_adapter=0):
        super().__init__()

        self.embeddings = copy.deepcopy(basemodel.bert.embeddings)
        
        self.blocks = nn.Sequential(*[
            Block(
            config = config,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            attn_drop=config.attention_probs_dropout_prob,
            LoRA_dim=LoRA_dim,  # LoRA Dimension
            prefix_dim=prefix_dim,  # Prefix Tuning
            adapter_dim = adapter_dim,
            drop_rate_LoRA=drop_rate_LoRA,
            drop_rate_adapter=drop_rate_adapter
            )
            for i in range(config.num_hidden_layers)])
        
        self.pool = pool
        # if pool:
        #     self.pooler = BertPooler(in_features=config.hidden_size, out_features=config.hidden_size)
        self.dropout = nn.Dropout(p=0.02, inplace=False)
        self.head = nn.Linear(in_features=config.hidden_size, out_features=num_classes, bias=True) if num_classes > 0 else nn.Identity()
        nn.init.normal_(self.head.weight, std=0.01)  
        nn.init.constant_(self.head.bias, 0.0)

        self.LoRA_dim = LoRA_dim
        self.adapter_dim = adapter_dim
        self.prefix_dim = prefix_dim
        self.embed_dim = config.hidden_size

        self.init_weights(weight_init)
        self.load_pretrained(basemodel)
        self.freeze_stages()

    def freeze_stages(self):  

        for name,param in self.embeddings.named_parameters():
            param.requires_grad = False

        for block in self.blocks:        
            for name,param in block.named_parameters():
                if 'adapter' not in name and 'prompt' not in name and 'LoRA' not in name and 'prefix' not in name and 'head' not in name and 'pool' not in name:
                    param.requires_grad = False


        total_para_nums = 0
        adapter_para_nums = 0
        LoRA_para_nums = 0
        vp_para_nums = 0
        head_para_nums = 0
        pool_para_nums = 0
        for name,param in self.named_parameters():
            if param.requires_grad:
                total_para_nums += param.numel()
                if 'adapter' in name:
                    adapter_para_nums += param.numel()
                elif 'LoRA' in name:
                    LoRA_para_nums += param.numel()
                elif 'prefix' in name:
                    vp_para_nums += param.numel()
                elif 'head' in name:
                    head_para_nums += param.numel()
                elif 'pool' in name:
                    pool_para_nums += param.numel()
                
                
        print('parameters:',total_para_nums,'adapter',adapter_para_nums,'LoRA',LoRA_para_nums,'prefix',vp_para_nums,'head',head_para_nums, 'pool', pool_para_nums)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        self.apply(_init_weights)

    @torch.jit.ignore()
    def load_pretrained(self, basemodel, prefix=''):
        _load_weights(self, basemodel, prefix)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)


    def set_sample_config(self, config: dict):

        # LoRA
        self.sample_DoRA_dim = config['lora_dim']

        # Adapter
        self.sample_s_adapter_dim = config['s_adapter_dim']
        self.sample_p_adapter_dim = config['p_adapter_dim']

        # prefix_tuning
        self.sample_prefix_dim = config['prefix_dim']

        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            blocks.set_sample_config(
                                     sample_DoRA_dim = self.sample_DoRA_dim[i],
                                     sample_prefix_dim = self.sample_prefix_dim[i],
                                     sample_s_adapter_dim = self.sample_s_adapter_dim[i],
                                     sample_p_adapter_dim = self.sample_p_adapter_dim[i]
                                    )

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                numels.append(module.calc_sampled_param_num())

        return sum(numels)

    def forward(self, input_ids, attention_mask=None, return_hidden_state=False):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
            
        if return_hidden_state:
            return x

        if self.pool:
            x = x.mean(dim=1)  
            # x = self.pooler(x)
            # cell_embedding = x[:,0]
            # x = cell_embedding.clone()
        x = self.dropout(x)
        x = self.head(x)
        return x

def _init_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if 'head' in name.lower():
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif jax_impl:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model, basemodel, prefix: str = ''):
    model.embeddings.word_embeddings.weight.copy_(basemodel.bert.embeddings.word_embeddings.weight)
    model.embeddings.position_embeddings.weight.copy_(basemodel.bert.embeddings.position_embeddings.weight)
    model.embeddings.token_type_embeddings.weight.copy_(basemodel.bert.embeddings.token_type_embeddings.weight)
    model.embeddings.LayerNorm.weight.copy_(basemodel.bert.embeddings.LayerNorm.weight)
    model.embeddings.LayerNorm.bias.copy_(basemodel.bert.embeddings.LayerNorm.bias)

    for i, block in enumerate(model.blocks):
        block.attn.query.weight.copy_(basemodel.bert.encoder.layer[i].attention.self.query.weight)
        block.attn.key.weight.copy_(basemodel.bert.encoder.layer[i].attention.self.key.weight)
        block.attn.value.weight.copy_(basemodel.bert.encoder.layer[i].attention.self.value.weight)
        block.attn.query.bias.copy_(basemodel.bert.encoder.layer[i].attention.self.query.bias)
        block.attn.key.bias.copy_(basemodel.bert.encoder.layer[i].attention.self.key.bias)
        block.attn.value.bias.copy_(basemodel.bert.encoder.layer[i].attention.self.value.bias)
        block.attn.query_LoRA_m.copy_(block.attn.query.weight.norm(p=2, dim=0))
        block.attn.value_LoRA_m.copy_(block.attn.value.weight.norm(p=2, dim=0))
        block.attn_output.linear.weight.copy_(basemodel.bert.encoder.layer[i].attention.output.dense.weight)
        block.attn_output.linear.bias.copy_(basemodel.bert.encoder.layer[i].attention.output.dense.bias)
        block.attn_output.LayerNorm.weight.copy_(basemodel.bert.encoder.layer[i].attention.output.LayerNorm.weight)
        block.attn_output.LayerNorm.bias.copy_(basemodel.bert.encoder.layer[i].attention.output.LayerNorm.bias)
        block.bertintermediate.weight.copy_(basemodel.bert.encoder.layer[i].intermediate.dense.weight)
        block.bertintermediate.bias.copy_(basemodel.bert.encoder.layer[i].intermediate.dense.bias)
        block.output.linear.weight.copy_(basemodel.bert.encoder.layer[i].output.dense.weight)
        block.output.linear.bias.copy_(basemodel.bert.encoder.layer[i].output.dense.bias)
        block.output.LayerNorm.weight.copy_(basemodel.bert.encoder.layer[i].output.LayerNorm.weight)
        block.output.LayerNorm.bias.copy_(basemodel.bert.encoder.layer[i].output.LayerNorm.bias)