

import torch
from torch.utils.data import DataLoader
from model.supernet import GeneFormer

from transformers import AutoTokenizer, BertForSequenceClassification, BertForTokenClassification, BertConfig
from Geneformer.tokenizer import TranscriptomeTokenizer

from datasets import load_from_disk
import os

from lib.collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from lib.config import cfg, update_config_from_file

def get_data(dataset_dir):
    import pickle
    with open('./Geneformer/token_dictionary_gc30M.pkl', "rb") as f:
        gene_token_dict = pickle.load(f)

    val_data=load_from_disk(os.path.join(dataset_dir,"train.dataset"))
    def remove_cols(data, cols_to_keep):
        other_cols = list(data.features.keys())
        other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
        data = data.remove_columns(other_cols)
        return data
    cols_to_keep = ["label", "input_ids", "length"]
    val_data = remove_cols(val_data, cols_to_keep)
    # for hcm
    import pandas as pd
    df = val_data.to_pandas()
    df['label'] = pd.factorize(df['label'])[0]
    val_data = val_data.from_pandas(df[['input_ids', 'length', 'label']]) # 只保留需要的列


    data_collator = DataCollatorForCellClassification(
        token_dictionary=gene_token_dict)

    data_loader_val = torch.utils.data.DataLoader(
        val_data, batch_size=40,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=False
    )

    return data_loader_val

def print_umap(nb_classes, resume, title, conf, dataset_dir, device):
    basemodel = BertForSequenceClassification.from_pretrained('../Geneformer/gf-12L-30M-i2048')
    config = BertConfig.from_pretrained('../Geneformer/gf-12L-30M-i2048/config.json')
    update_config_from_file(conf)
    model = GeneFormer(
        config=config, 
        basemodel=basemodel, 
        num_classes=nb_classes, 
        pool=True,
        weight_init='',
        LoRA_dim=cfg.SUPERNET.LORA_DIM,
        adapter_dim=cfg.SUPERNET.ADAPTER_DIM,
        prefix_dim=cfg.SUPERNET.PREFIX_DIM,
        drop_rate_LoRA=0.1,
        drop_rate_adapter=0.1
        )

    checkpoint = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    embeddings = []
    labels = []
    data_loader_val = get_data(dataset_dir)
    model.to(device)

    config = {'lora_dim':cfg.RETRAIN.LORA_DIM,'p_adapter_dim':cfg.RETRAIN.P_ADAPTER_DIM,'s_adapter_dim':cfg.RETRAIN.S_ADAPTER_DIM,'prefix_dim':cfg.RETRAIN.PREFIX_DIM,}
    model.set_sample_config(config=config)
    with torch.no_grad():
        for batch in data_loader_val:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels']
            cell_embeddings = model(input_ids, attention_mask,return_hidden_state=True)
            embeddings.append(cell_embeddings.detach().cpu().mean(dim=1).squeeze(1))
            labels.append(label)

    # import pickle
    # with open('./out/human_dcm_hcm/human_dcm_hcm_id_class_dict.pkl', "rb") as f:
    #     label_dict = pickle.load(f)
    #     print(label_dict)
    # labels = [label_dict[label_id] for label_id in labels]

    # 拼接所有batch的嵌入
    X_embedding = torch.cat(embeddings, dim=0).numpy()  # 形状=[n_cells, embed_dim]
    labels = torch.cat(labels, dim=0).numpy()  # 形状=[n_cells, embed_dim]
    import scanpy as sc
    import pandas as pd
    import numpy as np

    # 创建AnnData对象
    adata = sc.AnnData(X_embedding)  # 嵌入作为主矩阵

    adata.obs['cell_type'] = pd.Categorical(labels)  # 假设labels是细胞类型列表

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    sc.pl.umap(adata, color='cell_type',frameon=True, 
        palette="tab20",       
        title="scAutoFM"
    )
    # celltype_colors = {
    #     "T cell": "#4f3dd5",
    #     "B cell": "#8c564b",
    #     "CD8-positive, alpha-beta T cell": "#e377c2",
    #     "CD4-positive, alpha-beta T cell": "#1f77b4",
    #     "macrophage": "#2ca02c",
    #     "natural killer cell": "#9467bd",
    #     "endothelial cell": "#17becf",
    #     "monocyte": "#d62728",
    #     "non-classical monocyte": "#bcbd22",
    #     "neutrophil": "#7f7f7f",
    #     "plasma cell": "#aec7e8",
    #     "intermediate monocyte": "#ff7f0e",
    #     "kidney epithelial cell": "#ffbb78"
    # }
    # celltype_colors = {
    #     "fibroblast": "#7f7f7f",
    #     "T cell": "#1f77b4",
    #     "mast cell": "#e377c2",
    #     "endothelial cell": "#ff7f0e",
    #     "hepatocyte": "#8c564b",
    #     "erythrocyte": "#ff9896",
    #     "macrophage": "#2ca02c",
    #     "B cell": "#bcbd22",
    #     "monocyte": "#17becf",
    #     "natural killer cell": "#9467bd",
    #     "CD4-positive, alpha-beta T cell": "#1f77b4",
    #     "CD8-positive, alpha-beta T cell": "#e377c2",
    #     "hepatic stellate cell": "#c49c94",
    #     "myeloid cell": "#d62728",
    #     "neutrophil": "#c7c7c7",
    #     "myeloid dendritic cell": "#98df8a",
    #     "plasmacytoid dendritic cell": "#f7b6d2",
    #     "plasma cell": "#aec7e8",
    #     "mature NK T cell": "#c5b0d5",
    #     "classical monocyte": "#ffbb78",
    #     "non-classical monocyte": "#dbdb8d",
    #     "intermediate monocyte": "#ff9896",
    #     "intrahepatic cholangiocyte": "#9edae5",
    #     "hematopoietic precursor cell": "#393b79"
    # }
    # # 给 adata.obs['celltype'] 设置颜色
    # sc.tl.umap(adata)
    # # 然后绘图
    # sc.pl.umap(adata, color="cell_type", palette=celltype_colors,title="scAutoFM",frameon=True,legend_loc="none")

    fig = plt.gcf()  # Get Current Figure
    fig.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形

    # from sklearn.cluster import KMeans
    # from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # # 聚类
    # # n_clusters = len(set(labels))
    # # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # # pred_labels = kmeans.fit_predict(X_embedding)

    # # # 计算 ARI 和 NMI
    # # ari = adjusted_rand_score(labels, pred_labels)
    # # nmi = normalized_mutual_info_score(labels, pred_labels)

    # # print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    # # print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 计算 ARI / NMI（基于 embedding 的无监督聚类）
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    pred_labels = kmeans.fit_predict(X_embedding)
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    from collections import Counter
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    def compute_iLISI(embedding, batch_labels, k=30):
        """
        embedding: n_cells x n_features numpy array
        batch_labels: n_cells 长度的列表或 pd.Series，表示每个细胞的批次/患者ID
        k: 邻居数量
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
        distances, indices = nbrs.kneighbors(embedding)
        ilisi_scores = []
        for idx, neighbors in enumerate(indices):
            neighbors = neighbors[neighbors != idx]  # 去掉自身
            neighbor_batches = [batch_labels[i] for i in neighbors]
            counts = np.array(list(Counter(neighbor_batches).values()))
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            ilisi_scores.append(np.exp(entropy))  # iLISI = exp(Shannon entropy)
        return np.mean(ilisi_scores)

    def compute_ebm(embedding, batch_labels, k=30):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embedding)
        distances, indices = nbrs.kneighbors(embedding)
        ebm_scores = []
        for idx, neighbors in enumerate(indices):
            # 去掉自身
            neighbors = neighbors[neighbors != idx]
            neighbor_batches = [batch_labels[i] for i in neighbors]
            counts = np.array(list(Counter(neighbor_batches).values()))
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            ebm_scores.append(entropy)
        return np.mean(ebm_scores)
    # 或 patient ID
    import numpy as np
    import pandas as pd

    # embedding: n_cells x n_features 的 numpy array
    # batch_labels: 对应的批次/患者 ID

    # adata.obs['cell_type'] = pd.Categorical(labels)  # 或 patient ID
    # ilisi_mean = compute_iLISI(adata.obsm['X_umap'], adata.obs['cell_type']).mean()
    # ebm_score = compute_ebm(adata.obsm['X_umap'], adata.obs['cell_type'], k=30)
    # print(f"iLISI: {ilisi_mean:.4f}, EBM: {ebm_score:.4f}")

if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser('plot', add_help=False)
    parser.add_argument('--title', default='my_liver', type=str)
    parser.add_argument('--resume', default='./saves/subnet_liver/checkpoint.pth', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--dataset_dir', default='./out/liver', type=str)
    parser.add_argument('--nb_classes', default=24, type=int)
    parser.add_argument('--cfg',help='experiment configure file name',default='./experiments/scAutoFM/subnet/liver.yaml',type=str)
    args = parser.parse_args()
    print_umap(args.nb_classes, args.resume, args.title, args.cfg, args.dataset_dir, args.device)

