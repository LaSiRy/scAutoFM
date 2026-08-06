import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 激活数据（和你提供的一样）
data = {
    'Aorta': {
        'LoRA': [50, 50, 50, 10, 25, 0, 0, 0, 0, 0, 0, 0],
        'Adapter': [100, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Parallel Adapter': [50, 50, 50, 50, 5, 5, 50, 5, 100, 0, 50, 0],
        'Prefix': [10, 10, 10, 50, 25, 10, 25, 10, 5, 25, 0, 10],
    },
    'Liver': {
        'LoRA': [25, 25, 25, 50, 0, 0, 0, 0, 0, 0, 0, 0],
        'Adapter': [50, 50, 5, 50, 5, 100, 10, 5, 5, 0, 0, 0],
        'Parallel Adapter': [100, 100, 0, 10, 100, 5, 50, 0, 0, 0, 0, 0],
        'Prefix': [5, 5, 25, 10, 50, 50, 5, 25, 0, 0, 0, 0],
    },
    'Kidney': {
        'LoRA': [10, 5, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0],
        'Adapter': [0, 50, 5, 0, 100, 10, 0, 0, 0, 0, 0, 0],
        'Parallel Adapter': [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Prefix': [10, 25, 25, 0, 25, 25, 0, 0, 5, 0, 0, 0],
    },
    'Cardiomyopathy': {
        'LoRA': [50, 10, 5, 100, 50, 50, 0, 0, 0, 0, 0, 0],
        'Adapter': [5, 5, 50, 100, 10, 50, 0, 0, 0, 0, 0, 0],
        'Parallel Adapter': [100, 10, 10, 50, 10, 100, 50, 50, 100, 10, 5, 100],
        'Prefix': [25, 50, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
}

layers = [f'Layer {i+1}' for i in range(12)]

# 找到所有激活值的最大值，用于统一颜色刻度
all_values = []
for module_data in data.values():
    for values in module_data.values():
        all_values.extend(values)
vmax = max(all_values)

# 绘制综合图
fig, axes = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)

for ax, (dataset_name, module_data) in zip(axes, data.items()):
    df = pd.DataFrame(module_data, index=layers).T
    sns.heatmap(df, annot=False, cmap='YlOrRd', cbar=True, linewidths=0.5, ax=ax, vmax=vmax)
    ax.set_title(f'{dataset_name} Activation', fontsize=10)
    ax.set_ylabel('Module')
    ax.set_xlabel('Layer')

plt.savefig("All_Datasets_Activation.png", dpi=300, bbox_inches='tight')
plt.show()
