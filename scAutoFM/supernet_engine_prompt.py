import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
    precision_score, 
    recall_score
)
from lib import utils
import random
import time
import numpy as np
from lib.collator_for_classification import DataCollatorForGeneClassification
from pathlib import Path
from tqdm import tqdm
from lib.datasets import preprocess_classifier_batch
import pickle

def sample_configs(choices, is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_adapter == False and is_LoRA == False and is_prefix==False:
        lora_depth = random.choice(choices['lora_depth'])
        s_adapter_depth = random.choice(choices['adapter_depth'])
        p_adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        if lora_depth >= 0:
            config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        else:
            lora_depth = -lora_depth
            config['lora_dim'] = [0] * (depth - lora_depth) + [random.choice(choices['lora_dim']) for _ in range(lora_depth)]
        if s_adapter_depth >= 0:
            config['s_adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(s_adapter_depth)] + [0] * (depth - s_adapter_depth)
        else:
            s_adapter_depth = -s_adapter_depth
            config['s_adapter_dim'] = [0] * (depth - s_adapter_depth) + [random.choice(choices['adapter_dim']) for _ in range(s_adapter_depth)]
        if p_adapter_depth >= 0:
            config['p_adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(p_adapter_depth)] + [0] * (depth - p_adapter_depth)
        else:
            p_adapter_depth = -p_adapter_depth
            config['p_adapter_dim'] = [0] * (depth - p_adapter_depth) + [random.choice(choices['adapter_dim']) for _ in range(p_adapter_depth)]
        if prefix_depth >= 0:
            config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)
        else:
            prefix_depth = -prefix_depth
            config['prefix_dim'] = [0] * (depth - prefix_depth) + [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)]
    else:
        if is_adapter:
            config['s_adapter_dim'] = [choices['super_adapter_dim']] * (depth)
            config['p_adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['s_adapter_dim'] = [0] * (depth)
            config['p_adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def train(model: torch.nn.Module, criterion: torch.nn.Module,
          data_loader: Iterable, optimizer: torch.optim.Optimizer,
          task_type: str,
          device: torch.device, loss_scaler, batch_size: int, max_norm: float = None, 
          amp: bool = False, retrain_config=None):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Supernet training: '
    print_freq = 100
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)

    step = 0
    accumulation_steps = int(batch_size/4)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                if task_type == "gene":
                    outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, labels)
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            if task_type == "gene":
                outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss = loss / accumulation_steps
        if amp:
            loss_scaler.scale(loss).backward() 
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if amp:
                loss_scaler.unscale_(optimizer)
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 
                optimizer.step()
                    
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, 
                    task_type: str, nb_classes: int,
                    device: torch.device, epoch: int, loss_scaler, batch_size: int, max_norm: float = None, 
                    amp: bool = False, fp16: bool = False,choices=None, mode='super', retrain_config=None,is_adapter=False,is_LoRA=False,is_prefix=False):
    model.train()
    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
    
    step = 0
    # accumulation_steps = int(batch_size/4)
    accumulation_steps = 1

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if mode == 'super':
            # sample config
            config = sample_configs(choices=choices,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                if task_type == "gene":
                    outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, labels)
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            if task_type == "gene":
                outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss = loss / accumulation_steps
        if amp:
            loss_scaler.scale(loss).backward() 
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if amp:
                loss_scaler.unscale_(optimizer)
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 
                optimizer.step()
                    
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, task_type, device, batch_size=4, amp=False, fp16=False, choices=None, mode='super', retrain_config=None,is_adapter=False,is_LoRA=False,is_prefix=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    all_preds = []
    all_labels = []
    for batch in metric_logger.log_every(data_loader, 100, header):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
        if task_type == "cell":
            predictions = torch.argmax(outputs, dim=1)
            acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            batch_size = input_ids.shape[0]
            metric_logger.meters['acc'].update(acc, n=batch_size)
        all_preds += [outputs.to("cpu").detach()]
        all_labels += [labels.to("cpu").detach()]
    
    if task_type == "cell":
        metric_logger.synchronize_between_processes()
        print('* Acc {acc.global_avg:.3f}'
          .format(acc=metric_logger.acc))
        result =  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        preds = torch.cat(all_preds)
        last_dim = len(preds.shape) - 1
        all_preds = preds.reshape(-1, preds.shape[last_dim])
        labels = torch.cat(all_labels)
        all_labels = torch.flatten(labels)
        preds_label_paired = [
            item
            for item in list(zip(all_preds.tolist(), all_labels.tolist()))
            if item[1] != -100
        ]
        y_pred = [utils.vote(item[0]) for item in preds_label_paired]
        y_true = [item[1] for item in preds_label_paired]
        
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_true, y_pred)
        print('f1: ', f1, 'precision:', precision, 'recall: ', recall)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    else:
        preds = torch.cat(all_preds)
        last_dim = len(preds.shape) - 1
        all_preds = preds.reshape(-1, preds.shape[last_dim])
        labels = torch.cat(all_labels)
        all_labels = torch.flatten(labels)
        preds_label_paired = [
            item
            for item in list(zip(all_preds.tolist(), all_labels.tolist()))
            if item[1] != -100
        ]
        y_pred = [utils.vote(item[0]) for item in preds_label_paired]
        y_true = [item[1] for item in preds_label_paired]
        logits_list = [item[0] for item in preds_label_paired]
        y_score = [utils.get_softmax(item)[1] for item in logits_list]

        acc = accuracy_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        print('acc: ', acc, 'auc: ', roc_auc)
        result = {'acc': acc, 'auc': roc_auc, 'f1_score': f1, 'conf_matrix': conf_matrix, 'precision': precision, 'recall': recall}
        print(result)
        return result

# need to be simple
def train_cross_valid(model: torch.nn.Module, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, lr_scheduler,
                    device: torch.device, epoch: int, loss_scaler, batch_size: int, 
                    data_dir,num_workers,
                    pin_mem: bool = True, 
                    max_norm: float = 0, 
                    amp: bool = True, choices=None, mode='super', retrain_config=None,is_adapter=False,is_LoRA=False,is_prefix=False):
    accs = []
    aucs = []
    fpr = []
    tpr = []
    f1_scores = []
    conf_matrices = []

    for iter in range(0, 5):
        trainset=load_from_disk(f"{data_dir}/train_gene_labeled_ksplit{iter+1}.dataset")
        valset=load_from_disk(f"{data_dir}/valid_gene_labeled_ksplit{iter+1}.dataset")

        trainset = trainset.shuffle(seed=42)
            
        max_trainset_len = max(trainset.select([i for i in range(len(trainset))])["length"])
        max_valset_len = max(valset.select([i for i in range(len(valset))])["length"])
        dataset_train = preprocess_classifier_batch(trainset, "gene", max_trainset_len, label_name="labels")
        dataset_val = preprocess_classifier_batch(valset, "gene", max_valset_len, label_name="labels")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_collator = DataCollatorForGeneClassification(
            token_dictionary=gene_token_dict)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=4,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=32,
            sampler=sampler_val, num_workers=num_workers,
            pin_memory=pin_mem, drop_last=False
        )

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, "gene", 
                device, step, loss_scaler, batch_size, max_norm,
                amp=amp,choices=choices, mode=mode, retrain_config=retrain_config,
                is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)

        test_state = evaluate(data_loader_val, model, "gene", device, amp=amp,choices=choices, mode=mode, retrain_config=retrain_config, is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
        accs.append(test_state['acc'])
        aucs.append(test_state['auc'])
        fpr.append(test_state['fpr'])
        tpr.append(test_state['tpr'])
        f1_scores.append(test_state['f1_score'])
        conf_matrices.append(test_state['conf_matrix'])
    
    avg_acc = np.mean(accs)

    avg_roc_auc = np.mean(aucs)
    roc_auc_sd = np.std(aucs)

    avg_f1_score = np.mean(f1_scores) 
    overall_conf_matrix = np.sum(conf_matrices, axis=0)
    # Return a dictionary with the metrics for further analysis
    metrics = {
        'conf_matrix': overall_conf_matrix,
        'macro_f1': avg_f1_score,
        'acc':avg_acc,
        'all_roc_metrics': {
            'tpr':tpr,
            'fpr':fpr,
            'all_roc_auc': aucs,
            'roc_auc': avg_roc_auc,
            'roc_auc_sd': roc_auc_sd,
        }
    }
    print(metrics)
    return metrics