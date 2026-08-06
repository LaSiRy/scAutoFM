from lib import numpy_compat  # noqa: F401


import torch
from torch.utils.data import DataLoader
from model.supernet import GeneFormer
from model.supernet_moe import GeneFormer_MOE

from transformers import AutoTokenizer, BertForSequenceClassification, BertForTokenClassification, BertConfig
from Geneformer.tokenizer import TranscriptomeTokenizer

from datasets import load_from_disk
import os

from lib.collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from lib.config import cfg, update_config_from_file
from lib.datasets import preprocess_classifier_batch
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def get_data(dataset_dir):
    import pickle
    with open('./Geneformer/token_dictionary_gc30M.pkl', "rb") as f:
        gene_token_dict = pickle.load(f)

    val_data=load_from_disk(os.path.join(dataset_dir,"./data/gene/bivalent_vs_lys4_only/train_gene_labeled_ksplit3.dataset"))
    
    max_val_data_len = max(val_data.select([i for i in range(len(val_data))])["length"])
    dataset_val = preprocess_classifier_batch(val_data, "gene", max_val_data_len, label_name="labels")

    data_collator = DataCollatorForGeneClassification(
        token_dictionary=gene_token_dict)

    data_loader_val = torch.utils.data.DataLoader(
        val_data, batch_size=100,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=False
    )

    return data_loader_val

def print_umap(nb_classes, resume, title, conf, dataset_dir, device):
    basemodel = BertForSequenceClassification.from_pretrained('../Geneformer/gf-12L-30M-i2048')
    config = BertConfig.from_pretrained('../Geneformer/gf-12L-30M-i2048/config.json')
    update_config_from_file(conf)
    model = GeneFormer_MOE(
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
        i = 0
        for batch in data_loader_val:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels']
            outputs = model(input_ids, attention_mask,return_hidden_state=True)
            cell_embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
            mask = (label != -100) 
            labeled_gene_embeds = cell_embeddings[mask] 
            
            labeled_gene_labels = label[mask]
            embeddings.append(labeled_gene_embeds.detach().cpu())
            labels.append(labeled_gene_labels.detach().cpu())
            del input_ids, attention_mask, label, outputs, cell_embeddings, mask, labeled_gene_embeds, labeled_gene_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            i += 1
            if i > 10:
                break

    X_filtered = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    del embeddings
    gc.collect()

    # 计算无监督聚类指标（ARI / NMI）
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    pred_labels = kmeans.fit_predict(X_filtered)
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
   
    # 创建AnnData对象并使用与 baseline 一致的 UMAP 流程
    adata = sc.AnnData(X_filtered)
    y_str = np.array(['Dosage-sensitive TFs' if l == 0 else 'Dosage-insensitive TFs' for l in labels])
    adata.obs['gene_type'] = pd.Categorical(y_str)

    print("Computing PCA / neighbors / UMAP ...")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30, metric="cosine")
    sc.tl.umap(adata, min_dist=0.6, spread=1.4, random_state=42)

    # 与 baseline 同风格：JointGrid + marginal KDE
    plot_df = pd.DataFrame({
        "UMAP_1": adata.obsm["X_umap"][:, 0],
        "UMAP_2": adata.obsm["X_umap"][:, 1],
        "category": adata.obs["gene_type"],
    })
    color_dict = {"Dosage-sensitive TFs": "#FFD700", "Dosage-insensitive TFs": "#1E90FF"}

    with plt.rc_context({
        "axes.facecolor": "white",
        "axes.grid": False,
        "figure.facecolor": "white",
    }):
        g = sns.JointGrid(
            data=plot_df,
            x="UMAP_1",
            y="UMAP_2",
            hue="category",
            palette=color_dict,
            height=8,
            ratio=5,
            space=0.05,
        )
        g.plot_joint(
            sns.scatterplot,
            s=10,
            alpha=1.0,
            edgecolor=None,
            rasterized=True,
        )
        g.plot_marginals(
            sns.kdeplot,
            fill=True,
            alpha=0.28,
            common_norm=False,
        )

        for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.2)

        g.ax_marg_x.set_xticks([])
        g.ax_marg_x.set_yticks([])
        g.ax_marg_y.set_xticks([])
        g.ax_marg_y.set_yticks([])
        g.ax_marg_x.set_xlabel("")
        g.ax_marg_y.set_ylabel("")

        g.fig.suptitle(title, fontsize=14, y=1.02)
        g.ax_joint.legend(title=None, loc="upper right", frameon=True, fontsize=10)

        # plt.tight_layout()
        # g.savefig(f"{title}_density.png", dpi=300, bbox_inches="tight")
        # plt.close(g.fig)
        # print(f"Saved density figure to: {title}_density.png")


if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser('plot', add_help=False)
    parser.add_argument('--title', default='scAutoFM: Dosage sensitivity', type=str)
    parser.add_argument('--resume', default='./saves/supernet_bivalent_vs_lys4_only_split3_moe/checkpoint.pth', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dataset_dir', default='', type=str) #./out/bivalent_vs_no_methyl
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--cfg',help='experiment configure file name',default='./experiments/scAutoFM/subnet/bivalent_vs_lys4_only.yaml',type=str)
    args = parser.parse_args()
    print_umap(args.nb_classes, args.resume, args.title, args.cfg, args.dataset_dir, args.device)

# ./experiments/scAutoFM/subnet/bivalent_vs_no_methyl.yaml 2