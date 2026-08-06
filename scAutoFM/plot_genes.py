import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import seaborn as sns
from scipy import stats
import gseapy as gp
from gseapy.plot import barplot, dotplot
import requests
from io import StringIO
import re

# 设置绘图风格
plt.style.use('default')
sns.set_style("whitegrid")
# 假设数据已加载到DataFrame中，这里我们创建一个示例DataFrame
# 您应该用您的实际数据替换这部分

data = pd.read_csv('./out/out_perturb_stat/perturbation_fromNF.csv')

# 设置显著性阈值
fdr_threshold = 0.05

data['Sig_to_DCM'] = data['Goal_end_FDR'] < fdr_threshold
data['Sig_to_HCM'] = data['Alt_end_FDR_HCM'] < fdr_threshold

# 分类基因
data['Role'] = 'Non-significant'
data.loc[data['Sig_to_HCM'] & (data['Shift_to_alt_end_HCM'] > 0), 'Role'] = 'HCM Promoter'
data.loc[data['Sig_to_DCM'] & (data['Shift_to_goal_end'] > 0), 'Role'] = 'DCM Promoter'
data.loc[data['Sig_to_HCM'] & (data['Shift_to_alt_end_HCM'] < 0), 'Role'] = 'HCM Suppressor'
data.loc[data['Sig_to_DCM'] & (data['Shift_to_goal_end'] < 0), 'Role'] = 'DCM Suppressor'

# 1. 绘制韦恩图展示显著基因重叠
plt.figure(figsize=(8, 6))
sig_dcm_genes = set(data[data['Sig_to_DCM']]['Ensembl_ID'])
sig_hcm_genes = set(data[data['Sig_to_HCM']]['Ensembl_ID'])


venn = venn2([sig_dcm_genes, sig_hcm_genes], 
             set_labels=('DCM', 'HCM'))
venn_circles = venn2_circles([sig_dcm_genes, sig_hcm_genes], linestyle='dashed', linewidth=1)

# 设置颜色
for patch in venn.patches:
    patch.set_alpha(0.5)
venn.get_patch_by_id('10').set_color('skyblue')
venn.get_patch_by_id('01').set_color('lightcoral')
venn.get_patch_by_id('11').set_color('pink')

venn = venn2([sig_dcm_genes, sig_hcm_genes], 
             set_labels=('DCM', 'HCM'))

plt.title('Overlap of Genes Significant for DCM and HCM', fontsize=16)
plt.tight_layout()
plt.show()
fig = plt.gcf()
fig.savefig("perturb_venn.png", dpi=300, bbox_inches='tight')

# 2. 绘制基因效应方向图
plt.figure(figsize=(8, 6))

# 创建复合分类
data['Combined_Effect'] = 'Non-significant'
data.loc[data['Sig_to_DCM'] & data['Sig_to_HCM'], 'Combined_Effect'] = 'Both significant'
data.loc[data['Sig_to_DCM'] & ~data['Sig_to_HCM'], 'Combined_Effect'] = 'DCM significant'
data.loc[~data['Sig_to_DCM'] & data['Sig_to_HCM'], 'Combined_Effect'] = 'HCM significant'

# 为不同分类设置颜色
color_map = {
    'Both significant': '#DA70D6',   # orchid 淡紫色
    'DCM significant': '#87CEFA',           # light sky blue
    'HCM significant': '#FF7F50',           # coral 珊瑚红
    'Non-significant': '#D3D3D3'     # light gray
}

# 绘制散点图
# order = ['Non-significant', 'DCM significant', 'HCM significant', 'Both significant']

# for effect in order:
#     subset = data[data['Combined_Effect'] == effect]
#     plt.scatter(subset['Shift_to_goal_end'], subset['Shift_to_alt_end_HCM'], 
#                 c=color_map[effect], label=effect, alpha=0.7, s=100)

# # 添加标签
# top_genes = data.nsmallest(10, 'Goal_end_FDR')

# from adjustText import adjust_text

# texts = []
# for i, row in top_genes.iterrows():
#     texts.append(
#         plt.text(row['Shift_to_goal_end'], row['Shift_to_alt_end_HCM'], 
#                  row['Gene_name'], fontsize=9)
#     )

# # 自动调整标签，避免重叠
# adjust_text(texts, 
#             arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

# # 添加参考线
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
# plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)

# # 设置轴标签和标题
# plt.xlabel('Shift to DCM', fontsize=14)
# plt.ylabel('Shift to HCM', fontsize=14)
# plt.title('Gene Deletion Effects on DCM and HCM Transitions', fontsize=16)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # 调整布局
# plt.tight_layout()
# plt.show()
# fig = plt.gcf()
# fig.savefig("perturb_sscatter.png", dpi=300, bbox_inches='tight')

# 3. 功能富集分析
# 提取显著促进DCM和HCM的基因
dcm_promoter_genes = data[(data['Role'] == 'Promotes DCM')]['Gene_name'].tolist()
hcm_promoter_genes = data[(data['Role'] == 'Promotes HCM')]['Gene_name'].tolist()

print(f"Number of DCM promoter genes: {len(dcm_promoter_genes)}")
print(f"Number of HCM promoter genes: {len(hcm_promoter_genes)}")

# 进行GO富集分析
def run_enrichment_analysis(gene_list, description):
    """运行GO富集分析"""
    try:
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'],
                         description=description,
                         cutoff=0.5,
                         outdir=None)
        return enr
    except Exception as e:
        print(f"Enrichment analysis failed for {description}: {e}")
        return None

# 运行富集分析
dcm_enr = run_enrichment_analysis(dcm_promoter_genes, 'DCM_Promoter')
hcm_enr = run_enrichment_analysis(hcm_promoter_genes, 'HCM_Promoter')

# 4. 可视化富集分析结果
def plot_enrichment_results(enr_result, title, top_n=10):
    """绘制富集分析结果"""
    if enr_result is None or enr_result.results.empty:
        print(f"No enrichment results for {title}")
        return
    
    # 获取前top_n个最显著的结果
    top_results = enr_result.results.head(top_n)
    
    # 创建绘图
    plt.figure(figsize=(8, 6))
    
    # 创建条形图
    y_pos = np.arange(len(top_results))
    plt.barh(y_pos, -np.log10(top_results['Adjusted P-value']), 
             color='steelblue', alpha=0.7)
    
    # 添加标签
    plt.yticks(y_pos, top_results['Term'])
    plt.xlabel('-log10(Adjusted P-value)', fontsize=14)
    plt.title(f'Top Enriched Terms: {title}', fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    fig = plt.gcf()  # Get Current Figure
    fig.savefig(f"perturbation_rich_{title}.png", dpi=300, bbox_inches='tight')

# 绘制DCM和HCM的富集结果
if dcm_enr is not None:
    plot_enrichment_results(dcm_enr, 'DCM Promoter Genes')

if hcm_enr is not None:
    plot_enrichment_results(hcm_enr, 'HCM Promoter Genes')

# 5c. DCM基因的富集分析条形图（简化版）
if dcm_enr is not None and not dcm_enr.results.empty:
    top_dcm = dcm_enr.results.head(5)
    y_pos = np.arange(len(top_dcm))
    axes[1, 0].barh(y_pos, -np.log10(top_dcm['Adjusted P-value']), color='skyblue')
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([term[:40] + '...' if len(term) > 40 else term for term in top_dcm['Term']])
    axes[1, 0].set_xlabel('-log10(Adjusted P-value)')
    axes[1, 0].set_title('C. Top DCM Enriched Terms', fontsize=14)

# 5d. HCM基因的富集分析条形图（简化版）
if hcm_enr is not None and not hcm_enr.results.empty:
    top_hcm = hcm_enr.results.head(5)
    y_pos = np.arange(len(top_hcm))
    axes[1, 1].barh(y_pos, -np.log10(top_hcm['Adjusted P-value']), color='lightcoral')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([term[:40] + '...' if len(term) > 40 else term for term in top_hcm['Term']])
    axes[1, 1].set_xlabel('-log10(Adjusted P-value)')
    axes[1, 1].set_title('D. Top HCM Enriched Terms', fontsize=14)

plt.tight_layout()
plt.show()
fig = plt.gcf()  # Get Current Figure
fig.savefig(f"perturbation_final.png", dpi=300, bbox_inches='tight')