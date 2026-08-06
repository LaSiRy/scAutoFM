from lib import numpy_compat  # noqa: F401
import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib import utils
from supernet_engine_prompt import evaluate
import argparse
import os
import yaml

from mmengine.dist import init_dist

import model as models
from timm.models import load_checkpoint
from datasets import load_from_disk
from lib.datasets import preprocess_classifier_batch
from model.supernet import GeneFormer
import sys
from transformers import AutoTokenizer, BertForSequenceClassification, BertForTokenClassification, BertConfig

def get_args_parser():
    parser = argparse.ArgumentParser('Choose Optimal Subnet', add_help=False)
    parser.add_argument('--checkpoint_path', type=str, default='')    
    parser.add_argument('--assessment', type=str, default='auc') 
    return parser

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), list(cand_tuple[2 * depth + 1: 3 * depth + 1])

def main(args):
    checkpoint_path = args.checkpoint_path
    info = torch.load(checkpoint_path, map_location="cpu")

    memory = info['memory']
    candidates = info['candidates']
    vis_dict = info['vis_dict']
    keep_top_k = info['keep_top_k']
    
    best_assessment = 0
    total_params = 2
    for i, cand in enumerate(keep_top_k[50]):
        if vis_dict[cand][args.assessment] >= best_assessment:
            if vis_dict[cand]['params'] < total_params:
                best_model = cand
        else:
            break

    depth, lora_dim, adapter_dim, prefix_dim = decode_cand_tuple(best_model)
    print("depth: ", depth)
    print("lora_dim: ", lora_dim)
    print("adapter_dim: ", adapter_dim)
    print("prefix_dim: ", prefix_dim)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('choose optimal subnet', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
