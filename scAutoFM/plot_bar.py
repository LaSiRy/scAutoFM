import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.transforms import offset_copy

# data = {
#     # "Method": [
#     #     "Geneformer", "scGPT",
#     #     "LoRA-only", "Adapter-only", "Parallel Adapter-only", "Prefix-only", "scAutoFM"
#     # ],
#     # "Dosage sensitivity": [0.91, 0.95, 0.908, 0.928, 0.891, 0.888, 0.933],
#     # "Bivalent vs non-methylated": [0.93, 0.89, 0.84, 0.92, 0.73, 0.82, 0.96],
#     # "Bivalent vs Lys4-methylated": [0.88, 0.90, 0.95, 0.91, 0.946, 0.89, 0.96],
#     # "AVG": [0.90, 0.91, 0.90, 0.92, 0.86, 0.87, 0.95],
#     # "Dosage sensitivity param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.84],
#     # "Bivalent vs non-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.47],
#     # "Bivalent vs Lys4-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 1.64],
#     # "AVG param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.98]
#     "Method": [
#         "scGPT",
#         "Geneformer",
#         "GeneCompass",
#         "LoRA-only",
#         "Adapter-only",
#         "Parallel Adapter-only",
#         "Prefix-only",
#         "scAutoFM"
#     ],
#     # 4个组织的ACC数据，顺序对应methods
#     "Cardiomyopathy" :[0.92, 0.88, 0.88, 0.68, 0.46, 0.78, 0.84, 0.91],
#     "Kidney":        [0.84, 0.96, 0.94, 0.90, 0.58, 0.88, 0.86, 0.96],
#     "Aorta":       [0.95, 0.95, 0.96, 0.96, 0.95, 0.96, 0.95, 0.97],
#     "Liver":      [0.80, 0.89, 0.90, 0.86, 0.85, 0.88, 0.86, 0.89],
#     "AVG":    [0.8775, 0.92, 0.92, 0.85, 0.71, 0.875, 0.8775, 0.93],
#     "AVG param":[51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.80]
# }

# df = pd.DataFrame(data)
# sns.set(style="whitegrid", font_scale=1.1)

# tasks = [
#     # ("Dosage sensitivity", "Dosage sensitivity param"),
#     # ("Bivalent vs non-methylated", "Bivalent vs non-methylated param"),
#     # ("Bivalent vs Lys4-methylated", "Bivalent vs Lys4-methylated param"),
#     ("AVG", "AVG param")
# ]

# # 自定义等距映射函数
# bins = [0, 2, 8, 16, 32, 64, 128, 150]
# n_bins = len(bins) - 1
# vis_positions = np.linspace(0, 1, n_bins + 1)

# def map_to_equal_spacing(x):
#     x = np.array(x)
#     y = np.zeros_like(x, dtype=float)
#     for i in range(n_bins):
#         mask = (x >= bins[i]) & (x < bins[i + 1])
#         y[mask] = vis_positions[i] + (x[mask] - bins[i]) / (bins[i+1]-bins[i]) * (vis_positions[i+1]-vis_positions[i])
#     y[x >= bins[-1]] = vis_positions[-1]
#     return y

# for task, param_col in tasks:
#     fig, ax = plt.subplots(figsize=(6, 4))

#     x_vals = df[param_col]
#     y_vals = df[task]

#     # 映射横坐标到等距
#     x_mapped = map_to_equal_spacing(x_vals)

#     # 散点 + 颜色映射
#     sc = ax.scatter(x_mapped, y_vals, s=100, c=y_vals, cmap="coolwarm", edgecolors="black")

#     # 标签在右侧
#     x_offset = 0.01
#     for xi, yi, label in zip(x_mapped, y_vals, df["Method"]):
#         ax.text(xi, yi+ x_offset, label, ha='left', va='bottom', fontsize=8)

#     # 设置横纵坐标
#     ax.set_xlabel("Training parameters (M)", fontsize=12)
#     ax.set_ylabel("ACC", fontsize=12)
#     ax.set_ylim(0.7, 0.96)

#     # 设置自定义横坐标刻度
#     tick_positions = map_to_equal_spacing(bins)
#     ax.set_xticks(tick_positions)
#     ax.set_xticklabels([str(b) for b in bins])

#     # 颜色条
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("ACC", fontsize=12)

#     ax.set_title(f"Average" if task == "AVG" else f"{task}")
#     plt.tight_layout()
#     plt.savefig(f"{task}_scatter_1.png", dpi=300, bbox_inches='tight')
#     plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = {
    "Method": [
        "scGPT",
        "Geneformer",
        "GeneCompass",
        "LoRA-only",
        "Adapter-only",
        "Parallel Adapter-only",
        "Prefix-only",
        "scAutoFM"
    ],
    "Cardiomyopathy":[0.92, 0.88, 0.88, 0.68, 0.46, 0.78, 0.84, 0.91],
    "Kidney":        [0.84, 0.96, 0.94, 0.90, 0.58, 0.88, 0.86, 0.96],
    "Aorta":         [0.95, 0.95, 0.96, 0.96, 0.95, 0.96, 0.95, 0.97],
    "Liver":         [0.80, 0.89, 0.90, 0.86, 0.85, 0.88, 0.86, 0.89],
    "AVG":           [0.8775, 0.92, 0.92, 0.85, 0.71, 0.875, 0.8775, 0.93],
    "Cardiomyopathy param" :[51, 30, 103, 0.22, 0.22, 0.22, 0.16, 1.95],
    "Kidney param":        [51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.51],
    "Aorta param":         [51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.80],
    "Liver param":         [51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.92],
    "AVG param":     [51, 30, 103, 0.22, 0.22, 0.22, 0.16, 1.04]
    # "Method": [
    #     "Geneformer", "scGPT",
    #     "LoRA-only", "Adapter-only", "Parallel Adapter-only", "Prefix-only", "scAutoFM"
    # ],
    # "Dosage sensitivity": [0.91, 0.95, 0.908, 0.928, 0.891, 0.888, 0.933],
    # "Bivalent vs non-methylated": [0.93, 0.89, 0.84, 0.92, 0.73, 0.82, 0.96],
    # "Bivalent vs Lys4-methylated": [0.88, 0.90, 0.95, 0.91, 0.946, 0.89, 0.96],
    # "AVG": [0.90, 0.91, 0.90, 0.92, 0.86, 0.87, 0.95],
    # "Dosage sensitivity param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.84],
    # "Bivalent vs non-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.47],
    # "Bivalent vs Lys4-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 1.64],
    # "AVG param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.98]
}

color = {
    "GenePT": "#ff7f0e",
    "scGPT": "#98df8a",
    "Geneformer": "#2ca02c",
    "GeneCompass": "#aec7e8",
    "LoRA-only": "#d62728",
    "Adapter-only": "#ff9896",
    "Parallel Adapter-only": "#9467bd",
    "Prefix-only": "#c5b0d5",
    "scAutoFM": "#8c564b",
}

# # color = {
# #     "Gene2vec-RF": "#1f77b4",
# #     "Gene2vec-LR": "#aec7e8",
# #     "GenePT-RF": "#ff7f0e",
# #     "GenePT-LR": "#ffbb78",
# #     "Geneformer": "#2ca02c",
# #     "scGPT": "#98df8a",
# #     "LoRA-only": "#d62728",
# #     "Adapter-only": "#ff9896",
# #     "Parallel Adapter-only": "#9467bd",
# #     "Prefix-only": "#c5b0d5",
# #     "scAutoFM": "#8c564b",
# # }

df = pd.DataFrame(data)
sns.set(style="whitegrid", font_scale=1.5)

tasks = [
    # ("Cardiomyopathy", "Cardiomyopathy param"),
    # ("Kidney", "Kidney param"),
    ("Aorta", "Aorta param"),
    # ("Liver", "Liver param"),
    # ("AVG", "AVG param")
]
# tasks = [
#     # ("Dosage sensitivity", "Dosage sensitivity param"),
#     # ("Bivalent vs non-methylated", "Bivalent vs non-methylated param"),
#     # ("Bivalent vs Lys4-methylated", "Bivalent vs Lys4-methylated param"),
#     ("AVG", "AVG param")
# ]

# 自定义等距映射函数
# bins = [0, 2, 16, 32, 64, 128, 150, 200]
bins = [0, 2, 4, 8, 16, 32, 64, 128, 150, 200]
n_bins = len(bins) - 1
vis_positions = np.linspace(0, 1, n_bins + 1)

def map_to_equal_spacing(x):
    x = np.array(x)
    y = np.zeros_like(x, dtype=float)
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        y[mask] = vis_positions[i] + (x[mask] - bins[i]) / (bins[i+1]-bins[i]) * (vis_positions[i+1]-vis_positions[i])
    y[x >= bins[-1]] = vis_positions[-1]
    return y

for task, param_col in tasks:
    fig, ax = plt.subplots(figsize=(6, 4))

    x_vals = df[param_col]
    y_vals = df[task]

    # 映射横坐标到等距
    x_mapped = map_to_equal_spacing(x_vals)

    colors_list = [color.get(m, "#333333") for m in df["Method"]]
    sc = ax.scatter(x_mapped, y_vals, s=150, c=colors_list, edgecolors="black") 

    # 标签错开逻辑
    x_offset = 0.03  # 向右偏移
    y_offset = 0.009 # 上下错开
    used_x = {}

    # for xi, yi, label in zip(x_mapped, y_vals, df["Method"]):
    #     rounded_x = round(xi, 3)
    #     found_close = None
    #     for prev_x in used_x:
    #         if abs(rounded_x - prev_x) < 0.015:  # 横向接近阈值
    #             found_close = prev_x
    #             break

    #     if found_close is not None:
    #         shift = (len(used_x[found_close]) % 2) * 2 - 1  # 交替 ±1
    #         ax.text(xi + x_offset, yi + shift * y_offset, label,
    #                 ha='left', va='center', fontsize=9)
    #         used_x[found_close].append(label)
    #     else:
    #         ax.text(xi + x_offset, yi, label,
    #                 ha='left', va='center', fontsize=9)
    #         used_x[rounded_x] = [label]
    # 先收集所有相同横坐标的标签及其纵坐标
    from collections import defaultdict
    used_x = defaultdict(list)
    used_y = {}
    for xi, yi, label in zip(x_mapped, y_vals, df["Method"]):
        rounded_x = round(xi, 3)
        used_x[rounded_x].append((yi, label))

    # 对每个横坐标组进行处理
    y_offset = 0.004  # 每个标签错开的高度
    for rounded_x, items in used_x.items():
        # 按纵坐标排序
        items_sorted = sorted(items, key=lambda x: x[0])
        n = len(items_sorted)
        mid = n // 2
        y_count = defaultdict(int)
        # 将中间点对齐原始位置，向上和向下错开
        for i, (yi, label) in enumerate(items_sorted):
            shift = (i - mid) * y_offset
            count = y_count[yi]
            # if count % 2 == 0:
            #     shift = y_offset  # 上方
            # else:
            #     shift = -y_offset  # 下方
            # if abs(rounded_x - 0.542) < 1e-6:
            #     shift = -0.004
            #     ax.text(rounded_x -0.04, yi + shift, label,
            #             ha='left', va='center', fontsize=9)
            # else:
            if yi==0.88:
                shift = -0.002
                ax.text(rounded_x+ x_offset, yi + shift, label,
                        ha='left', va='center', fontsize=9)
            elif abs(rounded_x - 0.009) < 1e-6:
                shift = 0.002
                ax.text(rounded_x+ x_offset, yi + shift, label,
                        ha='left', va='center', fontsize=9)
            elif abs(rounded_x - 0.012) < 1e-6 and yi==0.95:
                shift = -0.002
                ax.text(rounded_x+ x_offset, yi + shift, label,
                        ha='left', va='center', fontsize=9)
            elif abs(rounded_x - 0.542) < 1e-6 and yi==0.95:
                shift = -0.005
                ax.text(rounded_x+ x_offset, yi + shift, label,
                        ha='center', va='center', fontsize=9)
            else:
                ax.text(rounded_x + x_offset, yi + shift, label,
                    ha='left', va='center', fontsize=9)
            y_count[yi] += 1
            print(rounded_x)


    # 设置横纵坐标
    ax.set_xlabel("Training parameters (M)", fontsize=12)
    ax.set_ylabel("ACC", fontsize=12)
    ax.set_ylim(0.92, 1.0)

    # 设置自定义横坐标刻度
    tick_positions = map_to_equal_spacing(bins)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(b) for b in bins])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_title("Average" if task == "AVG" else f"{task}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{task}_param.png", dpi=300, bbox_inches='tight')
    plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# # 数据
# data = {
#     # "Method": [
#     #     "Gene2vec-RF", "Gene2vec-LR", "GenePT-RF", "GenePT-LR",
#     #     "Geneformer", "scGPT",
#     #     "LoRA-only", "Adapter-only", "Parallel Adapter-only", "Prefix-only", "scAutoFM"
#     # ],
#     # "Dosage sensitivity": [0.86, 0.91, 0.92, 0.89, 0.91, 0.95, 0.908, 0.928, 0.891, 0.888, 0.933],
#     # "Bivalent vs non-methylated": [0.63, 0.66, 0.92, 0.91, 0.93, 0.89, 0.84, 0.92, 0.73, 0.82, 0.96],
#     # "Bivalent vs Lys4-methylated": [0.89, 0.91, 0.93, 0.94, 0.88, 0.90, 0.95, 0.91, 0.946, 0.89, 0.96],
#     # "AVG": [0.79, 0.82, 0.92, 0.91, 0.90, 0.91, 0.90, 0.92, 0.86, 0.87, 0.95],
#     # "Dosage sensitivity param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.84],
#     # "Bivalent vs non-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.47],
#     # "Bivalent vs Lys4-methylated param": [30, 31, 0.22, 0.22, 0.22, 0.16, 1.64],
#     # "AVG param": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.98]
#     "Method": [
#         "GenePT",
#         "scGPT",
#         "Geneformer",
#         "GeneCompass",
#         "LoRA-only",
#         "Adapter-only",
#         "Parallel Adapter-only",
#         "Prefix-only",
#         "scAutoFM"
#     ],
#     # 4个组织的ACC数据，顺序对应methods
#     "Cardiomyopathy" :[0.54, 0.92, 0.88, 0.88, 0.68, 0.46, 0.78, 0.84, 0.91],
#     "Kidney":        [0.84, 0.84, 0.96, 0.94, 0.90, 0.58, 0.88, 0.86, 0.96],
#     "Aorta":       [0.88, 0.95, 0.95, 0.96, 0.96, 0.95, 0.96, 0.95, 0.97],
#     "Liver":      [0.33, 0.80, 0.89, 0.90, 0.86, 0.85, 0.88, 0.86, 0.89],
#     "AVG_cell":    [0.65, 0.88, 0.92, 0.92, 0.85, 0.71, 0.88, 0.88, 0.93],
#     "train_params":[None, 51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.80]
# }

# df = pd.DataFrame(data)

# # 绘图风格
# sns.set(style="whitegrid")
# colors_list = [color.get(m, "#333333") for m in df["Method"]]
# palette = sns.color_palette(colors_list)

# # tasks = ["Dosage sensitivity", "Bivalent vs non-methylated", "Bivalent vs Lys4-methylated", "AVG"]
# tasks = ["Kidney",] #"Aorta", "Kidney", "Cardiomyopathy", "Liver", "AVG_cell"
# for task in tasks:
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sns.barplot(data=df, x="Method", y=task, palette=palette, ax=ax, width=0.6)

#     # x轴标签旋转
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.set_ylabel("ACC", fontsize=12)
#     ax.set_xlabel("")
#     ax.set_ylim(0.5, 1.0)
#     if task == "AVG":
#         ax.set_title(f"Average performance")
#     else:
#         ax.set_title(f"Performance on {task}")

#     # 在柱子上方标注数值
#     for p in ax.patches:
#         ax.annotate(f"{p.get_height():.2f}",
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha="center", va="bottom",
#                     fontsize=8, color="black", xytext=(0, 2),
#                     textcoords="offset points")

#     plt.tight_layout()
#     plt.savefig(f"{task}_bar.png", dpi=300, bbox_inches='tight')
#     plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# # 数据
# data = {
#     # "Method": [
#     #     "Geneformer", "scGPT",
#     #     "LoRA-only", "Adapter-only", "Parallel Adapter-only", "Prefix-only", "scAutoFM"
#     # ],
#     # "Dosage sensitivity": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.84],
#     # "Bivalent vs non-methylated": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.47],
#     # "Bivalent vs Lys4-methylated": [30, 31, 0.22, 0.22, 0.22, 0.16, 1.64],
#     # "AVG": [30, 31, 0.22, 0.22, 0.22, 0.16, 0.98]
#     "Method": [
#         "scGPT",
#         "Geneformer",
#         "GeneCompass",
#         "LoRA-only",
#         "Adapter-only",
#         "Parallel Adapter-only",
#         "Prefix",
#         "scAutoFM"
#     ],
#     "AVG":[51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.80]
# }
# df = pd.DataFrame(data)

# sns.set(style="whitegrid")
# palette = sns.color_palette("tab20", n_colors=len(df["Method"].unique()))

# tasks = ["AVG"]

# for task in tasks:
#     fig, (ax_top, ax_bottom) = plt.subplots(
#         2, 1, sharex=True, figsize=(6, 3),
#         gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08}
#     )

#     x = range(len(df["Method"]))
#     y = df[task]

#     # 绘制柱状图
#     ax_top.bar(x, y, color=palette, width=0.5)
#     ax_bottom.bar(x, y, color=palette, width=0.5)

#     # 设置 y 轴范围
#     ax_top.set_ylim(28, 105)  # 高段
#     ax_bottom.set_ylim(0, 1)  # 低段
#     ax_top.tick_params(axis='y', labelsize=6)
#     ax_bottom.tick_params(axis='y', labelsize=6)

#     # 添加标签
#     for i, val in enumerate(y):
#         if val >= 28:  # 在上面标高值
#             ax_top.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
#         else:  # 在下面标低值
#             ax_bottom.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

#     # 断轴符号
#     d = .015  # 断口的大小
#     grid_color = plt.rcParams['grid.color']  # 取当前网格线颜色
#     kwargs = dict(transform=ax_top.transAxes, color=grid_color, clip_on=False)

#     ax_top.plot((-d, +d), (-d, +d), **kwargs)        # 左断口
#     ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右断口

#     kwargs.update(transform=ax_bottom.transAxes)
#     ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
#     ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

#     # kwargs.update(transform=ax_bottom.transAxes)
#     # ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
#     # ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

#     # x 轴标签和标题
#     ax_bottom.set_xticks(x)
#     ax_bottom.set_xticklabels(df["Method"], rotation=45, ha="right")
#     ax_bottom.set_ylabel("Training parameters (M)", fontsize=12)
#     ax_bottom.yaxis.set_label_coords(-0.07, 0.9) 
#     fig.suptitle(f"Average training parameters")

#     plt.tight_layout()
#     plt.savefig(f"{task}_param.png", dpi=300, bbox_inches='tight')
#     plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# methods = [
#     "Geneformer", "scGPT",
#     "LoRA-only", "Adapter-only", "Parallel Adapter-only", "Prefix-only", "scAutoFM"
# ]
# training_params = np.array([30, 31, 0.22, 0.22, 0.22, 0.16, 0.98])
# avg_auc = np.array([0.90, 0.91, 0.90, 0.92, 0.86, 0.87, 0.95])

# y = np.arange(len(methods))
# bar_height = 0.4

# fig, ax = plt.subplots(figsize=(10, 6))

# left_width = 0.5
# right_width = 0.5

# training_params_norm = (training_params / training_params.max()) * left_width
# auc_norm = avg_auc * right_width

# ax.barh(y - bar_height/2, training_params_norm, height=bar_height, color='skyblue', label='Training Params (M)')
# ax.barh(y + bar_height/2, auc_norm, left=1 - auc_norm, height=bar_height, color='orange', label='Avg AUC')

# ax.set_yticks(y)
# ax.set_yticklabels(methods)
# ax.invert_yaxis()

# ax.set_xlim(0, 1)

# # 左侧刻度和标签
# left_ticks = np.linspace(0, left_width, 6)
# left_labels = np.round(np.linspace(0, training_params.max(), 6), 1)

# # 右侧刻度位置
# right_ticks = np.linspace(left_width, 1, 6)
# # 右侧标签要倒序显示，和柱子方向匹配（从右到左）
# right_labels = np.round(np.linspace(1, 0, 6), 2)

# # 合并刻度
# all_ticks = np.concatenate((left_ticks, right_ticks))
# all_labels = list(left_labels.astype(str)) + list(right_labels.astype(str))
# ax.set_xticks(all_ticks)
# ax.set_xticklabels(all_labels)

# # X轴标题，放坐标轴下方，左右两侧
# ax.annotate('Training Params (M)',
#             xy=(left_width/2, 0), xycoords=('axes fraction', 'axes fraction'),
#             xytext=(0, -30), textcoords='offset points',
#             ha='center', va='top', fontsize=12)

# ax.annotate('Avg AUC',
#             xy=(left_width + right_width/2, 0), xycoords=('axes fraction', 'axes fraction'),
#             xytext=(0, -30), textcoords='offset points',
#             ha='center', va='top', fontsize=12)

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# plt.title('Training Params (Left) and Avg AUC (Right) from Both Ends (Equal Width)', fontsize=14)
# plt.tight_layout()
# plt.show()

# plt.savefig("param_auc.png", dpi=300, bbox_inches='tight')


# import matplotlib.pyplot as plt
# import numpy as np

# methods = [
#     "scGPT",
#     "Geneformer",
#     "GeneCompass",
#     "LoRA-only",
#     "Adapter-only",
#     "Parallel Adapter-only",
#     "Prefix-only",
#     "scAutoFM"
# ]
# training_params = np.array([51, 30, 103, 0.22, 0.22, 0.22, 0.16, 0.80])
# avg_auc = np.array([0.88, 0.92, 0.92, 0.85, 0.71, 0.88, 0.88, 0.93])

# y = np.arange(len(methods))
# bar_height = 0.4

# fig, ax = plt.subplots(figsize=(10, 6))

# left_width = 0.5
# right_width = 0.5

# training_params_norm = (training_params / training_params.max()) * left_width
# auc_norm = avg_auc * right_width

# ax.barh(y - bar_height/2, training_params_norm, height=bar_height, color='skyblue', label='Training Params (M)')
# ax.barh(y + bar_height/2, auc_norm, left=1 - auc_norm, height=bar_height, color='orange', label='Avg ACC')

# ax.set_yticks(y)
# ax.set_yticklabels(methods)
# ax.invert_yaxis()

# ax.set_xlim(0, 1)

# # 左侧刻度和标签
# left_ticks = np.linspace(0, left_width, 6)
# left_labels = np.round(np.linspace(0, training_params.max(), 6), 1)

# # 右侧刻度位置
# right_ticks = np.linspace(left_width, 1, 6)
# # 右侧标签要倒序显示，和柱子方向匹配（从右到左）
# right_labels = np.round(np.linspace(1, 0, 6), 2)

# # 合并刻度
# all_ticks = np.concatenate((left_ticks, right_ticks))
# all_labels = list(left_labels.astype(str)) + list(right_labels.astype(str))
# ax.set_xticks(all_ticks)
# ax.set_xticklabels(all_labels)

# # X轴标题，放坐标轴下方，左右两侧
# ax.annotate('Training Params (M)',
#             xy=(left_width/2, 0), xycoords=('axes fraction', 'axes fraction'),
#             xytext=(0, -30), textcoords='offset points',
#             ha='center', va='top', fontsize=12)

# ax.annotate('Avg ACC',
#             xy=(left_width + right_width/2, 0), xycoords=('axes fraction', 'axes fraction'),
#             xytext=(0, -30), textcoords='offset points',
#             ha='center', va='top', fontsize=12)

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# plt.title('Training Params (Left) and Avg ACC (Right) from Both Ends (Equal Width)', fontsize=14)
# plt.tight_layout()
# plt.show()

# plt.savefig("param_acc.png", dpi=300, bbox_inches='tight')