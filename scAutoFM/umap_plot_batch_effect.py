import scanpy as sc
import matplotlib.pyplot as plt

# 加载 h5ad 文件
# adata = sc.read_h5ad('./out/human_dcm_hcm/human_dcm_hcm.h5ad')  # 替换为你的文件路径

# target_donors = ["P1422", "P1510", "P1539", "P1606", "P1702"]

# adata_filtered = adata[adata.obs["donor_id"].isin(target_donors)].copy()

# adata_filtered = sc.read_h5ad('../data/geneformer/cell/aorta/sample_aorta_data.h5ad')
# adata_filtered = sc.read_h5ad('../data/geneformer/cell/kidney/kidney.h5ad')
# print(adata_filtered)
adata_filtered = sc.read_h5ad('../data/geneformer/cell/liver/liver.h5ad')
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)
sc.pp.pca(adata_filtered, n_comps=50)
sc.pp.neighbors(adata_filtered, n_neighbors=15)

fig = plt.figure(figsize=(8, 6)) 
sc.tl.umap(adata_filtered)
fig = sc.pl.umap(
    adata_filtered,
    color="donor_id",         
    frameon=True,           
    legend_loc="none",   
    palette="tab20",       
    title="Original liver data"
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
# sc.pl.umap(adata_filtered, color="cell_type", palette=celltype_colors,title="Original kidney data",legend_loc="none",frameon=True)

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
# fig = plt.figure(figsize=(8, 6)) 
# sc.tl.umap(adata_filtered)
# sc.pl.umap(adata_filtered, color="cell_type", palette=celltype_colors,title="Original liver data",legend_loc="none",frameon=True)

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
ilisi_mean = compute_iLISI(adata_filtered.obsm['X_umap'], adata_filtered.obs['donor_id'])
ebm_score = compute_ebm(adata_filtered.obsm['X_umap'], adata_filtered.obs['donor_id'], k=30)
print(f"iLISI: {ilisi_mean:.4f}, EBM: {ebm_score:.4f}")

fig = plt.gcf()  # Get Current Figure
fig.savefig("liver_ori_batch.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形

