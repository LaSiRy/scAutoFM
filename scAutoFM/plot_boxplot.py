import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('./out/out_perturb_stat/perturbation_fromNF_embedding.csv')
# data = pd.read_csv('./out/out_perturb_stat/perturb_human.csv')
# data = pd.read_csv('../baseline/out_perturb_stat/perturb_human_embedding.csv')
# data = pd.read_csv('../baseline/out_perturb_stat/perturb_human.csv')

cardiomyopathy_genes = {"MYH7", "TTN", "GATA4", "LMNA", "MYBPC3", "GATA4"}
structural_heart_disease_genes = {"TBX5","ZFPM2","SCN5A","ACTC1","NKX2-5",}#"TBX5",
hyperlipidaemia_genes = {"CD36", "BCO2", "PCSK6", "LPL"}

def classify_gene(row):
    gene = row["Gene"]
    if gene in cardiomyopathy_genes:
        return "Cardiomyopathy"
    elif gene in structural_heart_disease_genes:
        return "Structural heart disease"
    elif gene in hyperlipidaemia_genes:
        return "Hyperlipidaemia"
    else:
        return "Unclassified"

data["Group"] = data.apply(classify_gene, axis=1)

# === 2. 过滤掉不显著的基因 ===
# data = data[data["Sig"] == 1]
# === 3. 过滤掉未分类的基因 ===
data = data[data["Group"] != "Unclassified"]
# === 4. 绘制箱线图 ===
plt.figure(figsize=(8, 6))
# plt.xlim(0, 1) 
data['Cosine_sim'] = 1 -abs(data['Cosine_sim'])
sns.boxplot(data=data, x='Cosine_sim', y='Group', palette='Set2',showfliers=False)


# 样式美化
plt.title("Cardiomyocyte embeddings of scAutoFM", fontsize=14)
plt.ylabel("In silico deleted genes")
plt.xlabel("Cosine Similarity")
plt.xticks(rotation=15)

fig = plt.gcf()  # Get Current Figure
fig.savefig("perturbation.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形
plt.tight_layout()
plt.show()