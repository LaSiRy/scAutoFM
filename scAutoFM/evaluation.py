import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
from timm.scheduler import create_scheduler
# from transformers import get_scheduler
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, PolynomialLR, SequentialLR

from lib.datasets import preprocess_classifier_batch
from supernet_engine_prompt import train_one_epoch, evaluate
from lib import utils
from lib.config import cfg, update_config_from_file
from lib.collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)

from datasets import load_from_disk
from model.supernet import GeneFormer
from model.supernet_moe import GeneFormer_MOE
from transformers import AutoTokenizer, BertForSequenceClassification, BertForTokenClassification, BertConfig, get_cosine_schedule_with_warmup
from Geneformer.tokenizer import TranscriptomeTokenizer
import model as models
import os
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',default='./experiments/scAutoFM/supernet/supernet-B_prompt.yaml', required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='geneformer', type=str,
                        help='Name of model to train')
    parser.add_argument('--task_type', default='cell', type=str,
                        help='task_type: cell or gene')
    parser.add_argument('--task_name', default='', type=str)
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'vp','retrain','search'], help='mode of AutoFormer')
    # parser.add_argument('--input-size', default=224, type=int)
    # parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')
    parser.add_argument('--num-warmup-steps', type=int, default=500,
                        help='warm up steps of lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=5, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=3, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=1, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--renb_class', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Dataset parameters
    # parser.add_argument('--data-path', default='./data/imagenet/', type=str,
    #                     help='dataset path')
    # parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    # parser.add_argument('--inat-category', default='name',
    #                     choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
    #                     type=str, help='semantic granularity')
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--prepared_dataset_path', default=None, type=str,
                        help='dataset file path')
    parser.add_argument('--gene_class_path', default=None, type=str,
                        help='gene_class file path')
    parser.add_argument('--label_name', default="cell_type", type=str)
    parser.add_argument('--h5ad_file', default="", type=str)
    

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    # parser.set_defaults(amp=True)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--is_adapter', action='store_true')
    parser.add_argument('--is_LoRA', action='store_true')
    parser.add_argument('--is_prefix', action='store_true')

    parser.add_argument('--no_aug', action='store_true')

    parser.add_argument('--val_interval', default=5, type=int, help='validataion interval')

    parser.add_argument('--drop_rate_LoRA', type=float, default=0.1)
    parser.add_argument('--drop_rate_prompt', type=float, default=0.1)
    parser.add_argument('--drop_rate_adapter', type=float, default=0.1)

    parser.add_argument('--few-shot-seed', type=int, default=0)
    parser.add_argument('--few-shot-shot', type=int, default=2)

    parser.add_argument('--inception',action='store_true')

    # dataset
    parser.add_argument('--custom-dataset', action='store_true')
    parser.add_argument('--prepare-dataset', action='store_true')
    parser.add_argument('--h5ad-file', default=None, type=str, help='h5ad file location')
    parser.add_argument('--dataset_dir', default=None, type=str, help='directory where datasets store')
    parser.add_argument('--id-class-dict-file', default=None, type=str)
    
    parser.add_argument('--dataset_save_dir', default='./out', type=str)
    parser.add_argument('--dataset_id', default='1', type=str)
            
    return parser

def get_dataset(args):
    import pickle
    with open('../Geneformer/geneformer/token_dictionary_gc30M.pkl', "rb") as f:
        gene_token_dict = pickle.load(f)
    if args.custom_dataset:
        if args.prepare_dataset:
            from lib.datasets import data_process
            h5ad_file = args.h5ad_file
            args.nb_classes = data_process(h5ad_file, args.dataset_dir, data_num=None, val_size=0.1, test_size=0.1, label_name=args.label_name, random_state=42)
        import os
        train_data=load_from_disk(os.path.join(args.dataset_dir, "train.dataset"))
        val_data=load_from_disk(os.path.join(args.dataset_dir,"valid.dataset"))

        def remove_cols(data, cols_to_keep):
            other_cols = list(data.features.keys())
            other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
            data = data.remove_columns(other_cols)
            return data
        cols_to_keep = ["label", "input_ids", "length"]

        train_data = remove_cols(train_data, cols_to_keep)
        val_data = remove_cols(val_data, cols_to_keep)

        train_data = train_data.shuffle(seed=42) 
        # if len(train_data) > 50000:
        #     train_data = train_data.select(range(50000)) 

        dataset_labels = train_data["label"]
        label_counts = Counter(dataset_labels)

        weights = [1.0 / label_counts[label] for label in dataset_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        data_collator = DataCollatorForCellClassification(
            token_dictionary=gene_token_dict)

        data_loader_train = torch.utils.data.DataLoader(
            train_data,
            batch_size=4,
            collate_fn=data_collator,
            sampler=sampler,
        )

        data_loader_val = torch.utils.data.DataLoader(
            val_data, batch_size=32,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=False
        )

        return data_loader_train, data_loader_val

    elif args.task_type == "gene":
        trainset=load_from_disk(f"{args.dataset_dir}/train_gene_labeled_ksplit{args.dataset_id}.dataset")
        valset=load_from_disk(f"{args.dataset_dir}/valid_gene_labeled_ksplit{args.dataset_id}.dataset")

        trainset = trainset.shuffle(seed=42) 
        # select_data = int(10000)
        # trainset = trainset.select(range(select_data)) 
            
        max_trainset_len = max(trainset.select([i for i in range(len(trainset))])["length"])
        max_valset_len = max(valset.select([i for i in range(len(valset))])["length"])
        dataset_train = preprocess_classifier_batch(trainset, args.task_type, max_trainset_len, label_name="labels")
        dataset_val = preprocess_classifier_batch(valset, args.task_type, max_valset_len, label_name="labels")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_collator = DataCollatorForGeneClassification(
            token_dictionary=gene_token_dict)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=6,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=32,
            sampler=sampler_val, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )
        return data_loader_train, data_loader_val

def main(args):
    if args.task_type == "cell":
        basemodel = BertForSequenceClassification.from_pretrained('../Geneformer/gf-12L-30M-i2048')
    elif args.task_type == "gene":
        basemodel = BertForTokenClassification.from_pretrained('../Geneformer/gf-12L-30M-i2048')
        args.nb_classes=2 
    config = BertConfig.from_pretrained('../Geneformer/gf-12L-30M-i2048/config.json')
    update_config_from_file(args.cfg)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    # 创建数据集
    # from lib.datasets import data_process
    # h5ad_file = '/home/user3/lsy/llmtime/scAutoFM/data/h5ad/0a839c4b-10d0-4d64-9272-684c49a2c8ba/b50b6b99-fd44-4a6d-9ca8-b5b3479eabbd.h5ad'
    # output_dir = "/home/user3/lsy/llmtime/scAutoFM/data/data"
    # args.nb_classes = data_process(args.dataset_path, args.output_dir)
    if args.custom_dataset or args.task_type=="gene":
        data_loader_train, data_loader_val = get_dataset(args)

    print(f"Creating SuperNet")
    print(cfg)

    if args.mode == 'retrain' and "RETRAIN" in cfg:
        model = GeneFormer(
            config=config, 
            basemodel=basemodel, 
            num_classes=args.nb_classes, 
            pool=False if args.task_type == "gene" else True,
            weight_init='',
            LoRA_dim=cfg.SUPERNET.LORA_DIM,
            adapter_dim=cfg.SUPERNET.ADAPTER_DIM,
            prefix_dim=cfg.SUPERNET.PREFIX_DIM,
            drop_rate_LoRA=args.drop_rate_LoRA,
            drop_rate_adapter=args.drop_rate_adapter
            )
        retrain_config = {'lora_dim':cfg.RETRAIN.LORA_DIM,'s_adapter_dim':cfg.RETRAIN.S_ADAPTER_DIM,'p_adapter_dim':cfg.RETRAIN.P_ADAPTER_DIM,'prefix_dim':cfg.RETRAIN.PREFIX_DIM,}
        
    # else:
    # model = GeneFormer_MOE(
    #     config=config, 
    #     basemodel=basemodel, 
    #     num_classes=args.nb_classes, 
    #     pool=False if args.task_type == "gene" else True,
    #     weight_init='',
    #     LoRA_dim=cfg.SUPERNET.LORA_DIM,
    #     adapter_dim=cfg.SUPERNET.ADAPTER_DIM,
    #     prefix_dim=cfg.SUPERNET.PREFIX_DIM,
    #     drop_rate_LoRA=args.drop_rate_LoRA,
    #     drop_rate_adapter=args.drop_rate_adapter
    #     )

    choices = {'depth': cfg.SUPERNET.DEPTH,
               'super_LoRA_dim':cfg.SUPERNET.LORA_DIM,
               'super_adapter_dim':cfg.SUPERNET.ADAPTER_DIM,
               'super_prefix_dim':cfg.SUPERNET.PREFIX_DIM,
               'lora_dim':cfg.SEARCH_SPACE.LORA_DIM,
               'adapter_dim':cfg.SEARCH_SPACE.ADAPTER_DIM,
               'prefix_dim':cfg.SEARCH_SPACE.PREFIX_DIM,
               'lora_depth':cfg.SEARCH_SPACE.LORA_DEPTH,
               'adapter_depth':cfg.SEARCH_SPACE.ADAPTER_DEPTH,
               'prefix_depth':cfg.SEARCH_SPACE.PREFIX_DEPTH,
               }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False) #
        model.load_state_dict(checkpoint['model'], strict=False)

    if args.fp16:
        model.half()

    model_without_ddp = model
    model.to(device)
    teacher_model = None
    teacher_loss = None

    model_ema = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    # with open(output_dir / "config.yaml", 'w') as f:
    #     f.write(args_text)

    test_stats = evaluate(data_loader_val, model, args.task_type, device,  mode = args.mode, retrain_config=retrain_config, is_adapter=args.is_adapter,is_LoRA=args.is_LoRA,is_prefix=args.is_prefix)
    with open(f"./metrics/scAuto/metrics.pkl", "wb") as f:
        pickle.dump(test_stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
