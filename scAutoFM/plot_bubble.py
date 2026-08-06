import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# 数据准备
# ----------------------------
data = []

# Kidney
kidney = {
    "kidney epithelial cell": 9278, "CD8-positive, alpha-beta T cell": 537,
    "B cell": 494, "macrophage": 339, "CD4-positive, alpha-beta T cell": 299,
    "natural killer cell": 201, "endothelial cell": 101, "monocyte": 54,
    "non-classical monocyte": 27, "neutrophil": 22, "T cell": 17,
    "plasma cell": 5, "intermediate monocyte": 2
}
for k, v in kidney.items():
    data.append(["Kidney", k, v])

# Liver
liver = {
    "hepatocyte": 7414, "macrophage": 3232, "endothelial cell": 1774,
    "monocyte": 1721, "CD8-positive, alpha-beta T cell": 1290,
    "erythrocyte": 980, "mature NK T cell": 808, "plasma cell": 766,
    "natural killer cell": 745, "neutrophil": 739,
    "intrahepatic cholangiocyte": 665, "intermediate monocyte": 557,
    "classical monocyte": 326, "CD4-positive, alpha-beta T cell": 291,
    "hepatic stellate cell": 223, "B cell": 163, "fibroblast": 161,
    "non-classical monocyte": 113, "myeloid cell": 81,
    "myeloid dendritic cell": 56, "hematopoietic precursor cell": 41,
    "T cell": 32, "mast cell": 28, "plasmacytoid dendritic cell": 8
}
for k, v in liver.items():
    data.append(["Liver", k, v])

# Aorta
aorta = {
    "T cell": 3788, "macrophage dendritic cell": 2690, "smooth muscle cell type 1": 947, "fibroblast": 770,
    "natural killer cell": 515, "smooth muscle cell type 2": 294, "mesenchymal stem cell": 193, "endothelial cell": 166,
    "plasma cell": 141, "B cell": 63, "mast cell": 45
}
for k, v in aorta.items():
    data.append(["Aorta", k, v])

# Cardiomyopathy
# cardio = {"Non-failing heart": 63682, "Hypertrophic Cardiomyopathy": 58680, "Dilated Cardiomyopathy": 36107}
# for k, v in cardio.items():
#     data.append(["Cardiomyopathy", k, v])

# 转换为 DataFrame
df = pd.DataFrame(data, columns=["Dataset", "CellType", "Count"])

# 计算每个 cell type 总数，用于右侧标注
cell_totals = df.groupby("CellType")["Count"].sum().reset_index()

# 按总数排序，方便展示
cell_totals = cell_totals.sort_values("Count", ascending=False)
df["CellType"] = pd.Categorical(df["CellType"], categories=cell_totals["CellType"], ordered=True)

# ----------------------------
# 绘图
# ----------------------------
fig, ax = plt.subplots(figsize=(5,8))

scatter = ax.scatter(
    x=df["Dataset"], y=df["CellType"],
    s=df["Count"]/5,   # 调整点大小比例
    c=pd.factorize(df["CellType"])[0],  # 每个 cell_type 一个颜色
    cmap="tab20", alpha=0.8
)

# 右侧标注 cell 总数
for i, row in cell_totals.iterrows():
    ax.text(2.2, row["CellType"], row["Count"], 
            va="center", fontsize=9, color="black")

ax.set_xlabel("Dataset")
ax.set_ylabel("Cell Type")
ax.set_title("Overview of Cell Types across Datasets")

# 调整横轴标签
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
plt.savefig("bubble.png", dpi=300, bbox_inches='tight')
