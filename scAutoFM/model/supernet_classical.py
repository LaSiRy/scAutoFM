import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
import math
import copy
from model.module.adapter_super import AdapterSuper
from timm.models.layers import trunc_normal_, lecun_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.02, proj_drop=0., LoRA_dim=1024, prefix_dim=1024, drop_rate_LoRA=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.LoRA_dim = LoRA_dim
        self.prefix_dim = prefix_dim
        
        if LoRA_dim > 0:
            # LoRA:
            self.q_LoRA_a = nn.Linear(dim, LoRA_dim, bias=False)
            self.v_LoRA_a = nn.Linear(dim, LoRA_dim, bias=False)
            self.q_LoRA_b = nn.Linear(LoRA_dim, dim, bias=False)
            self.v_LoRA_b = nn.Linear(LoRA_dim, dim, bias=False)
            nn.init.xavier_uniform_(self.q_LoRA_a.weight)
            nn.init.xavier_uniform_(self.v_LoRA_a.weight)
            nn.init.zeros_(self.q_LoRA_b.weight)
            nn.init.zeros_(self.v_LoRA_b.weight)
        
        if prefix_dim > 0:
            self.prefix_tokens_key = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            self.prefix_tokens_value = nn.Parameter(torch.zeros(1, prefix_dim, dim))
            
            nn.init.xavier_uniform_(self.prefix_token_gate)
            
            nn.init.normal_(self.prefix_tokens_key, mean=0.0, std=0.02)
            nn.init.normal_(self.prefix_tokens_value, mean=0.0, std=0.02)


        self.LoRA_drop = nn.Dropout(p=drop_rate_LoRA)
        drop_rate_prefix = drop_rate_LoRA
        self.prefix_drop = nn.Dropout(p=drop_rate_prefix)

    def set_sample_config(self, sample_DoRA_dim,sample_prefix_dim):
        self.sample_DoRA_dim = sample_DoRA_dim
        self.LoRA_identity = False
        if self.sample_DoRA_dim == 0:
            self.LoRA_identity = True 
        else:
            self.q_LoRA_a_weight = self.q_LoRA_a.weight[:self.sample_DoRA_dim,:]
            self.v_LoRA_a_weight = self.v_LoRA_a.weight[:self.sample_DoRA_dim,:]

            self.q_LoRA_b_weight = self.q_LoRA_b.weight[:,:self.sample_DoRA_dim]
            self.v_LoRA_b_weight = self.v_LoRA_b.weight[:,:self.sample_DoRA_dim]

        self.sample_prefix_dim = sample_prefix_dim
        self.prefix_identity = False
        if self.sample_prefix_dim == 0:
            self.prefix_identity = True 
        else:
            self.prefix_weight_key = self.prefix_tokens_key[:,:self.sample_prefix_dim,:]
            self.prefix_weight_value = self.prefix_tokens_value[:,:self.sample_prefix_dim,:]
        
    def calc_sampled_param_num(self):
        if self.sample_DoRA_dim == 0:
            return 0
        else:
            return self.q_LoRA_a_weight.numel() + self.v_LoRA_a_weight.numel() + self.q_LoRA_b_weight.numel() + self.v_LoRA_b_weight.numel()
       
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # .reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.LoRA_identity == False:
            q_delta = F.linear(self.LoRA_drop(x), self.q_LoRA_a_weight)
            q_delta = F.linear(q_delta, self.q_LoRA_b_weight)
            v_delta = F.linear(self.LoRA_drop(x), self.v_LoRA_a_weight)
            v_delta = F.linear(v_delta, self.v_LoRA_b_weight)

            q = ((q+q_delta)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = ((v+v_delta)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        else:
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.prefix_identity == False:  

            prefix_k = self.prefix_weight_key.expand(B, -1, -1)
            prefix_v = self.prefix_weight_value.expand(B, -1, -1)

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
    def __init__(self, hidden_size, output_size, layer_norm_eps, adapter_dim, drop_rate_adapter, drop_rate=0.02):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.LayerNorm = nn.LayerNorm(output_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(drop_rate)

        self.serial_adapter = AdapterSuper(
            embed_dims=output_size,
            reduction_dims=adapter_dim,
            drop_rate_adapter=drop_rate_adapter
        )
        self.parallel_adapter = AdapterSuper(
            embed_dims=output_size,
            reduction_dims=adapter_dim,
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
                 drop_path=0., LoRA_dim=1024, adapter_dim=1024, prefix_dim=1024, drop_rate_LoRA=0, drop_rate_adapter=0):
        super().__init__()

        self.attn = Attention(config.hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              LoRA_dim=LoRA_dim, prefix_dim=prefix_dim, drop_rate_LoRA=drop_rate_LoRA)
        self.attn_output = BertSelfOutput(hidden_size=config.hidden_size, output_size=config.hidden_size, layer_norm_eps=config.layer_norm_eps)

        # DropPath for stochastic depth
        self.bertintermediate = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.intermediate_act_fn = nn.ReLU()

        self.output = BertOutput(hidden_size=config.intermediate_size, output_size=config.hidden_size, layer_norm_eps=config.layer_norm_eps, adapter_dim=adapter_dim, drop_rate_adapter=drop_rate_adapter)


    def set_sample_config(self, sample_DoRA_dim=None,sample_s_adapter_dim=None,sample_p_adapter_dim=None,sample_prefix_dim=None):

        self.attn.set_sample_config(sample_DoRA_dim=sample_DoRA_dim,sample_prefix_dim=sample_prefix_dim)
        self.sample_s_adapter_dim = sample_s_adapter_dim
        self.sample_p_adapter_dim = sample_p_adapter_dim
        self.output.serial_adapter.set_sample_config(sample_embed_dim=self.sample_s_adapter_dim)
        self.output.parallel_adapter.set_sample_config(sample_embed_dim=self.sample_p_adapter_dim)

    def calc_sampled_param_num(self):
        return 0

    def forward(self, x, attention_mask):
        attention_output = self.attn_output(self.attn(x, attention_mask), x)
        intermediate_output = self.intermediate_act_fn(self.bertintermediate(attention_output))
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class GeneFormer(nn.Module):
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