import matplotlib.pyplot as plt


iterations = list(range(1, 13))
# aorta
# lora = [50, 50, 50, 10, 25, 0, 0, 0, 0, 0, 0, 0]
# s_adapter = [100, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# p_adapter = [50, 50, 50, 50, 5, 5, 50, 5, 100, 0, 50, 0]
# apt = [10, 10, 10, 50, 25, 10, 25, 10, 5, 25, 0, 10]

# liver
# lora = [25, 25, 25, 50, 0, 0, 0, 0, 0, 0, 0, 0]
# s_adapter = [50, 50, 5, 50, 5, 100, 10, 5, 5, 0, 0, 0]
# p_adapter = [100, 100, 0, 10, 100, 5, 50, 0, 0, 0, 0, 0]
# apt = [5, 5, 25, 10, 50, 50, 5, 25, 0, 0, 0, 0]

# kidney
# lora = [10, 5, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0]
# s_adapter = [0, 50, 5, 0, 100, 10, 0, 0, 0, 0, 0, 0]
# p_adapter = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# apt = [10, 25, 25, 0, 25, 25, 0, 0, 5, 0, 0, 0]

# human
# lora = [50, 10, 5, 100, 50, 50, 0, 0, 0, 0, 0, 0]
# s_adapter = [5, 5, 50, 100, 10, 50, 0, 0, 0, 0, 0, 0]
# p_adapter = [100, 10, 10, 50, 10, 100, 50, 50, 100, 10, 5, 100]
# apt = [25, 50, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# # 创建图像
# plt.figure(figsize=(6, 4))

# # 绘制折线
# plt.plot(iterations, lora, 'o-', label='DoRA')
# plt.plot(iterations, s_adapter, 'd-', label='Adapter')
# plt.plot(iterations, p_adapter, 'g+-', label='Parallel Adapter')
# plt.plot(iterations, apt, 'v-', label='APT')

# # 设置标题和轴标签
# plt.title('Cardiomyopathy', fontsize=16)
# plt.xlabel('Layer(l)', fontsize=12)
# plt.ylabel('configuration', fontsize=12)

# # 设置刻度和网格
# plt.xticks(range(1, 13))
# plt.yticks([0,5,10,25,50,75,100])
# plt.grid(True, linestyle='-', alpha=0.5)

# # 图例
# plt.legend()

# # 显示图像
# fig = plt.gcf()  # Get Current Figure
# fig.savefig("human_arch.png", dpi=300, bbox_inches='tight')
# plt.close()  # 关闭图形
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 模块名 & 激活值
modules = ['APT', 'LoRA', 'S-Adapter', 'P-Adapter']  # 从上到下
colors = {
    'LoRA': '#66c2a5',
    'S-Adapter': '#fc8d62',
    'P-Adapter': '#8da0cb',
    'APT': '#e78ac3'
}
num_layers = 12

# 模块激活强度（每个列表长度为12）
lora      = [50, 50, 50, 10, 25, 0, 0, 0, 0, 0, 0, 0]
s_adapter = [100, 50, 50, 0,  0,  0, 0, 0, 0, 0, 0, 0]
p_adapter = [50, 50, 50, 50, 5,  5, 50, 5, 100, 0, 50, 0]
apt       = [10, 10, 10, 50, 25, 10, 25, 10, 5, 25, 0, 10]

module_values = {
    'LoRA': lora,
    'S-Adapter': s_adapter,
    'P-Adapter': p_adapter,
    'APT': apt
}

# 图设置
fig, ax = plt.subplots(figsize=(num_layers * 1.1, 3))
ax.set_xlim(0, num_layers)
ax.set_ylim(0, 1)
ax.axis('off')

# 每个 block 宽度和每个模块高度比例
block_width = 0.8
module_height = 1 / len(modules)
max_val = 100  # 最大激活值用于归一化高度

# 绘图
for layer_idx in range(num_layers):
    x_start = layer_idx

    # 画每个模块
    for mod_idx, module in enumerate(modules):
        y_base = mod_idx * module_height
        val = module_values[module][layer_idx]
        if val > 0:
            h = module_height * (val / max_val)  # 高度归一化
            rect = patches.Rectangle(
                (x_start + (1 - block_width)/2, y_base),  # 居中
                block_width,
                h,
                facecolor=colors[module],
                edgecolor='black',
                linewidth=0.3
            )
            ax.add_patch(rect)

    # 层标签
    ax.text(x_start + 0.5, -0.05, f"Layer {layer_idx}", ha='center', va='top', fontsize=8)

# 可选图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label=m, markerfacecolor=c, markersize=10)
    for m, c in colors.items()
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

plt.tight_layout()
plt.show()
fig = plt.gcf()  # Get Current Figure
fig.savefig("try.png", dpi=300, bbox_inches='tight')
