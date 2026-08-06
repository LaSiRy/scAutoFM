import scanpy as sc
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
from collections import Counter, defaultdict
from lib.utils import load_and_filter, filter_by_dict, validate_and_clean_cols, flatten_list, label_classes
from geneformer import Classifier
from datasets import Dataset, load_from_disk

# 假设你已经有了处理好的数据集 padded_batch
class MyDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def split_data(adata, val_size=0.1, test_size=0.1, label_name="cell_type", random_state=42):
    
    train_idx, temp_idx = train_test_split(np.arange(adata.n_obs), test_size=(val_size + test_size), random_state=random_state)
    if test_size > 0:
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (val_size + test_size), random_state=random_state)
        train_adata = adata[train_idx, :]
        val_adata = adata[val_idx, :]
        test_adata = adata[test_idx, :]
    else:
        val_idx = temp_idx
        train_adata = adata[train_idx, :]
        val_adata = adata[val_idx, :]
        test_adata = None    
    
    return train_adata, val_adata, test_adata

def dataset(adata, data_num=None, val_size=0.1, test_size=0.1, label_name="cell_type", random_state=42):
    unique_cell_types = adata.obs[label_name].nunique()

    if "gene_ids" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["gene_ids"]        
    else:
        adata.var["ensembl_id"] = adata.var.index
    adata.obs["n_counts"] = adata.X.sum(axis=1)

    if data_num is not None and data_num < adata.n_obs:
        np.random.seed(random_state)  # 设置随机种子，保证可复现
        selected_indices = np.random.choice(adata.n_obs, data_num, replace=False)  # 随机选择 data_num 个索引
        adata = adata[selected_indices, :]

    train_adata, val_adata, test_adata = split_data(adata, val_size, test_size, label_name, random_state)

    # 打印拆分后的数据
    print(f"Train set size: {train_adata.n_obs} cells")
    print(f"Valid set size: {val_adata.n_obs} cells")

    return train_adata, val_adata, test_adata, unique_cell_types



def split_data_by_ids(adata, attr_key, train_id, valid_id, test_id):
    """
    根据给定的细胞 ID 列表拆分数据集为训练集、验证集和测试集。
    
    """
    
    # 获取训练集和测试集细胞 ID
    train_adata = adata[adata.obs[attr_key].isin(train_id), :].copy()
    valid_adata = adata[adata.obs[attr_key].isin(valid_id), :].copy()
    test_adata = adata[adata.obs[attr_key].isin(test_id), :].copy()

    return train_adata, valid_adata, test_adata

# def data_process(h5ad_file, output_dir, attr_key, train_id, valid_id, test_id, label_name="cell_type", random_state=42):
#     adata = sc.read(h5ad_file)
#     unique_cell_types = adata.obs[label_name].nunique()
    
#     if "gene_ids" in adata.var.columns:
#         adata.var["ensembl_id"] = adata.var["gene_ids"]
#     else:
#         adata.var["ensembl_id"] = adata.var.index
#     adata.obs["n_counts"] = adata.X.sum(axis=1)

#     train_adata, valid_adata, test_adata = split_data_by_ids(adata, attr_key, train_id, valid_id, test_id)
#     # write in train and test
#     import os
    
#     train_path = os.path.join(output_dir, "train")
#     valid_path = os.path.join(output_dir, "val")
#     test_path = os.path.join(output_dir, "test")
    
#     os.makedirs(train_path, exist_ok=True)
#     os.makedirs(valid_path, exist_ok=True)
#     os.makedirs(test_path, exist_ok=True)

#     train_adata.write(os.path.join(output_dir, "train", "train.h5ad"))
#     valid_adata.write(os.path.join(output_dir, "val", "val.h5ad"))
#     test_adata.write(os.path.join(output_dir, "test", "test.h5ad"))

#     #tokenize
#     from Geneformer.tokenizer import TranscriptomeTokenizer

#     # custom_attr_name_dict={"cell_type": "cell_type", "tissue_type": "tissue_type"}

#     tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={f"{label_name}": f"{label_name}"}, model_input_size=2048, special_token=False, nproc=4)
#     file_format = "h5ad" 

#     # 调用 tokenize_data 函数
#     tokenizer.tokenize_data(
#         file_format=file_format,
#         data_directory=train_path,
#         output_directory = output_dir,
#         output_prefix = "train"
#     )

#     tokenizer.tokenize_data(
#         file_format=file_format,
#         data_directory=valid_path,
#         output_directory = output_dir,
#         output_prefix = "val"
#     )
        
#     tokenizer.tokenize_data(
#         file_format=file_format,
#         data_directory=test_path,
#         output_directory = output_dir,
#         output_prefix = "test"
#     )

#     import shutil
#     shutil.rmtree(os.path.join(output_dir,"train"))
#     shutil.rmtree(os.path.join(output_dir,"val"))
#     shutil.rmtree(os.path.join(output_dir,"test"))
#     return unique_cell_types

def data_process(h5ad_file, output_dir, data_num=50000, val_size=0.1, test_size=0.1, label_name="cell_type", random_state=42):
    adata = sc.read(h5ad_file)
    train_adata, val_adata, test_adata, unique_cell_types = dataset(adata, data_num, val_size, test_size, label_name, random_state)
    # write in train and test
    import os
    
    train_path = os.path.join(output_dir, "train")
    valid_path = os.path.join(output_dir, "val")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)

    train_adata.write(os.path.join(output_dir, "train", "train.h5ad"))
    val_adata.write(os.path.join(output_dir, "val", "val.h5ad"))

    #tokenize
    from Geneformer.tokenizer import TranscriptomeTokenizer

    # custom_attr_name_dict={"cell_type": "cell_type", "tissue_type": "tissue_type"}

    tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={f"{label_name}": f"{label_name}"}, model_input_size=2048, special_token=False, nproc=4)
    file_format = "h5ad" 

    # 调用 tokenize_data 函数
    tokenizer.tokenize_data(
        file_format=file_format,
        data_directory=train_path,
        output_directory = output_dir,
        output_prefix = "train"
    )

    tokenizer.tokenize_data(
        file_format=file_format,
        data_directory=valid_path,
        output_directory = output_dir,
        output_prefix = "valid"
    )

    if test_size > 0:
        test_path = os.path.join(output_dir, "test")
        os.makedirs(test_path, exist_ok=True)
        test_adata.write(os.path.join(output_dir, "test", "test.h5ad"))
        tokenizer.tokenize_data(
            file_format=file_format,
            data_directory=test_path,
            output_directory = output_dir,
            output_prefix = "test"
        )

    # import shutil
    # shutil.rmtree(os.path.join(output_dir,"train"))
    # shutil.rmtree(os.path.join(output_dir,"val"))
    # if test_size > 0:
    #     shutil.rmtree(os.path.join(output_dir,"test"))
    return unique_cell_types

import os
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

def preprocess_classifier_batch(cell_batch, task_type, max_len, label_name="cell_type"):
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    # load token dictionary (Ensembl IDs:token)
    from Geneformer.tokenizer import TOKEN_DICTIONARY_FILE
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    def pad_label_example(example):
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=gene_token_dict.get("<pad>"),
        )
        if task_type == "cell":
            example[label_name] = np.pad(
                example[label_name],
                (0, max_len - len(example["input_ids"])),
                mode="constant",
                constant_values=-100,
            )
        else:
            example[label_name] = np.pad(
                example[label_name],
                (0, max_len - len(example[label_name])),
                mode="constant",
                constant_values=-100,
            )
        example["attention_mask"] = (
            example["input_ids"] != gene_token_dict.get("<pad>")
        ).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    input_data_batch = torch.tensor(padded_batch["input_ids"])
    attn_msk_batch = torch.tensor(padded_batch["attention_mask"])
    label_batch = padded_batch[label_name]
    if task_type == "cell":
        label_encoder = LabelEncoder()
        label_indices = label_encoder.fit_transform(label_batch)
        label_batch = torch.tensor(label_indices, dtype=torch.long)
    else:
        label_batch = torch.tensor(label_batch, dtype=torch.long) 
    dataset = MyDataset(input_data_batch, attn_msk_batch, label_batch)

    return dataset

def prepare_gene_data(
    input_data_file,
    gene_class_dict,
    gene_token_dict = None,
    rare_threshold = 0,
    nproc = 4,
    num_crossval_splits = 1,
    split_id_dict = None,
    gene_balance = False,
):  

    empty_classes = []
    for k, v in gene_class_dict.items():
        if len(v) == 0:
            empty_classes += [k]
    if len(empty_classes) > 0:
        print(
            f"Class(es) {empty_classes} did not contain any genes in the token dictionary."
        )
        raise
    insuff_classes = [k for k, v in gene_class_dict.items() if len(v) < 5]
    if (num_crossval_splits > 0) and (len(insuff_classes) > 0):
        print(
            f"Insufficient # of members in class(es) {insuff_classes} to (cross-)validate."
        )
        raise
    data = load_from_disk(input_data_file)

    # convert classes to numerical labels and save as id_class_dict
    # of note, will label all genes in gene_class_dict
    # if (cross-)validating, genes will be relabeled in column "labels" for each split
    # at the time of training with Classifier.validate
    data, id_class_dict = label_classes(
        "gene", data, gene_class_dict, nproc
    )

    if split_id_dict is not None:
        data_dict = dict()
        data_dict["train"] = filter_by_dict(
            data, {split_id_dict["attr_key"]: split_id_dict["train"]}, nproc
        )
        data_dict["test"] = filter_by_dict(
            data, {split_id_dict["attr_key"]: split_id_dict["test"]}, nproc
        )
        return data_dict["train"], data_dict["test"], id_class_dict
        
    else:
        return data, id_class_dict