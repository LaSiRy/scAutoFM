import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv('./out/out_perturb_stat/perturb_human.csv')
# data = pd.read_csv('../baseline/out_perturb_stat/perturb_human.csv')

# 过滤 Sig == 1
sig_data = data[(data["Goal_end_FDR"] < 0.05) | (data["Alt_end_FDR_HCM"] < 0.05)]

# 只取第四象限 (x > 0, y < 0)
sig_data = sig_data[(sig_data["Shift_to_goal_end"] > 0) & 
                    (sig_data["Shift_to_alt_end_HCM"] < 0)]

# 绘制 KDE 等高线 (只画线条, 不填充)
plt.figure(figsize=(6,6))
ax = sns.kdeplot(
    data=sig_data,
    x="Shift_to_goal_end",
    y="Shift_to_alt_end_HCM",
    fill=False,        # ❌ 不填充
    cmap="plasma",     # 线条颜色映射
    levels=15,         # 等高线层数
    cbar=True          # ✅ 加颜色条
)

cbar = ax.collections[0].colorbar
cbar.set_label("Density →")       # 设置标签
cbar.set_ticks([])   

# 标签
plt.xlabel("Shift towards non-failing →", fontsize=12)
plt.ylabel("← Shift away from dilated cardiomyopathy", fontsize=12)
plt.title("Distribution of candidate therapeutic targets", fontsize=14)

# 限制坐标范围（只保留第四象限）
plt.xlim(0, sig_data["Shift_to_goal_end"].max()*1.1)
plt.ylim(-0.05,0)

plt.tight_layout()
plt.show()


fig = plt.gcf()  # Get Current Figure
fig.savefig("perturbation_KDE.png", dpi=300, bbox_inches='tight')
