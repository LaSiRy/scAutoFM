# use to convert .h5ad data to .dataset data
import os
from collections import Counter
from Geneformer.tokenizer import TranscriptomeTokenizer
from datasets import load_from_disk
import scanpy as sc
import pickle


def label_classes(data, nproc):
    label_set = set(data["label"])

    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    id_class_dict = {v: k for k, v in class_id_dict.items()}
    
    def classes_to_ids(example):
        example["label"] = class_id_dict[example["label"]]
        return example

    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict

def filter_by_dict(data, filter_data, nproc):
    for key, value in filter_data.items():

        def filter_data_by_criteria(example):
            return example[key] in value

        data = data.filter(filter_data_by_criteria, num_proc=nproc)
    if len(data) == 0:
        print("No cells remain after filtering. Check filtering criteria.")
        raise
    return data

def remove_rare(data, rare_threshold, label, nproc):
    if rare_threshold > 0:
        total_cells = len(data)
        label_counter = Counter(data[label])
        nonrare_label_dict = {
            label: [k for k, v in label_counter if (v / total_cells) > rare_threshold]
        }
        data = filter_by_dict(data, nonrare_label_dict, nproc)
    return data

def prepare_data(h5ad_path, output_directory, output_prefix, custom_attr_name_dict, label, rare_threshold, attr_key, split_id_dict=None):
    adata = sc.read(h5ad_path)
    if "gene_ids" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["gene_ids"]
    else:
        adata.var["ensembl_id"] = adata.var.index
    adata.obs["n_counts"] = adata.X.sum(axis=1)
    data_path = os.path.join(output_directory, f"{output_prefix}.h5ad")
    adata.write(data_path)

    tokenizer = TranscriptomeTokenizer(custom_attr_name_dict=custom_attr_name_dict, model_input_size=2048, special_token=False, nproc=4)
    file_format = "h5ad" 

    # 调用 tokenize_data 函数
    tokenizer.tokenize_data(
        file_format=file_format,
        data_directory=output_directory,
        output_directory = output_directory,
        output_prefix = output_prefix
    )

    data_path = os.path.join(output_directory, f"{output_prefix}.dataset")
    data = load_from_disk(data_path)
    data = remove_rare(data, rare_threshold, label, nproc=16)
    data = data.rename_column(label, "label")

    data, id_class_dict = label_classes(data, 16)
    if split_id_dict is not None:
        train_data = filter_by_dict(
            data, {attr_key: split_id_dict["train"]}, 16
        )

        valid_data = filter_by_dict(
            data, {attr_key: split_id_dict["valid"]}, 16
        )
        
        test_data = filter_by_dict(
            data, {attr_key: split_id_dict["test"]}, 16
        )
    else:
        split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1}
        data_dict = data.train_test_split(
            test_size=split_sizes["valid"]+split_sizes["test"],
            seed=42,
        )
        train_data = data_dict["train"]
        val_test_data = data_dict["test"]

        data_dict = train_val_data.train_test_split(
            test_size=split_sizes["test"]/(split_sizes["valid"]+split_sizes["test"]),
            seed=42,
        )
        val_data = data_dict["train"]
        test_data = data_dict["test"]
                   
    
    data_train_path = os.path.join(output_directory, f"{output_prefix}_train.dataset")
    data_valid_path = os.path.join(output_directory, f"{output_prefix}_valid.dataset")
    data_test_path = os.path.join(output_directory, f"{output_prefix}_test.dataset")

    train_data.save_to_disk(str(data_train_path))
    valid_data.save_to_disk(str(data_valid_path))
    test_data.save_to_disk(str(data_test_path))

    id_class_output_path = os.path.join(output_directory, f"{output_prefix}_label_id.pkl")
    with open(id_class_output_path, "wb") as f:
        pickle.dump(id_class_dict, f)

def split_h5ad_by_id(h5ad_path, attr_key, output_directory, split_id_dict=None):
    import numpy as np
    adata = sc.read(h5ad_path)
    from sklearn.model_selection import train_test_split
    if split_id_dict is not None:
        train_mask = adata.obs[attr_key].isin(split_id_dict["train"])
        eval_mask = adata.obs[attr_key].isin(split_id_dict["valid"])
        test_mask = adata.obs[attr_key].isin(split_id_dict["test"])

        train_data = adata[train_mask].copy()
        valid_data = adata[eval_mask].copy()
        test_data = adata[test_mask].copy()
    else:
        split_sizes = {"train": 0.8, "valid": 0.1, "test": 0.1}
    
        train_data, temp_data = train_test_split(
            adata,
            test_size=split_sizes["valid"] + split_sizes["test"],
            random_state=42
        )
        
        valid_data, test_data = train_test_split(
            temp_data,
            test_size=split_sizes["test"] / (split_sizes["valid"] + split_sizes["test"]),
            random_state=42
        )

    train_data.write(os.path.join(output_directory, "train.h5ad"))
    valid_data.write(os.path.join(output_directory, "valid.h5ad"))
    test_data.write(os.path.join(output_directory, "test.h5ad"))

if __name__ == '__main__':
    # h5ad_path = './out/human_dcm_hcm/human_dcm_hcm.h5ad'
    # output_directory = './out/human_dcm_hcm'
    # output_prefix = "human_dcm_hcm"
    # custom_attr_name_dict = {"donor_id": "donor_id", "disease": "disease"}
    # rare_threshold = 0
    # attr_key = "donor_id"
    # label = "disease"
    
    # train_ids = ["P1447", "P1600", "P1462", "P1558", "P1300", "P1508", "P1358", "P1678", "P1561", "P1304", "P1610", "P1430", "P1472", "P1707", "P1726", "P1504", "P1425", "P1617", "P1631", "P1735", "P1582", "P1722", "P1622", "P1630", "P1290", "P1479", "P1371", "P1549", "P1515"]
    # eval_ids = ["P1422", "P1510", "P1539", "P1606", "P1702"]
    # test_ids = ["P1437", "P1516", "P1602", "P1685", "P1718"]

    # split_id_dict = {"train": train_ids, "valid": eval_ids, "test":test_ids}
    # prepare_data(h5ad_path, output_directory, output_prefix, custom_attr_name_dict, label, rare_threshold, attr_key, split_id_dict)

    # h5ad_path = '../data/geneformer/cell/kidney.h5ad'
    # output_directory = './out/kidney'
    # output_prefix = "kidney"
    # custom_attr_name_dict = {"donor_id": "donor_id", "cell_type": "cell_type"}
    # rare_threshold = 0
    # attr_key = "donor_id"
    # label = "cell_type"

    # prepare_data(h5ad_path, output_directory, output_prefix, custom_attr_name_dict, label, rare_threshold, attr_key)
