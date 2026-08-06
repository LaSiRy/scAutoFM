import matplotlib.pyplot as plt


iterations = list(range(1, 11))
# aorta
method1 = [96.46, 96.46, 96.56, 96.67, 96.67, 96.68, 96.68, 96.68, 96.68, 96.68]
method2 = [96.46, 96.46, 96.46, 96.56, 96.56, 96.56, 96.56, 96.56, 96.56, 96.56]
# human
# method1 = [85.52, 87.57, 87.57, 88.05, 88.05, 88.05, 88.05, 88.05, 88.05, 88.05]
# method2 = [85.52, 86.06, 86.06, 86.06, 86.06, 86.06, 86.06, 86.06, 86.06, 86.06]
# kidney
# method1 = [89.35, 90.23, 91.12, 91.71, 91.71, 91.71, 91.82, 91.82, 91.82, 91.82]
# method2 = [89.35, 89.64, 90.23, 90.23, 90.23, 90.23, 90.23, 90.23, 90.23, 90.23]
# liver
# method1 = [88.68, 89.87, 89.87, 89.87, 89.91, 89.91, 90.27, 90.36, 90.36, 90.36]
# method2 = [88.68, 88.68, 89.82, 89.82, 89.82, 89.82, 89.82, 89.82, 89.82, 89.82]

# 创建图像
plt.figure(figsize=(6, 4))

# 绘制折线
plt.plot(iterations, method1, 'o--', label='scPEFTNAS')
plt.plot(iterations, method2, 'd--', label='Random Search')
# plt.plot(iterations, method3, 'g+-', label='Method 3')
# plt.plot(iterations, method4, 'v--', label='Method 4')

# 设置标题和轴标签
plt.title('Aorta', fontsize=16)
plt.xlabel('Iteration(T)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)

# 设置刻度和网格
plt.xticks(range(1, 11))
# plt.yticks([89.0, 89.5, 90.0, 90.5, 91.0, 91.5, 92.0])
plt.yticks([96.40, 96.45, 96.50, 96.55, 96.60, 96.65, 96.70])
plt.grid(True, linestyle='--', alpha=0.5)

# 图例
plt.legend()

# 显示图像
fig = plt.gcf()  # Get Current Figure
fig.savefig("search_aorta.png", dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形
plt.tight_layout()
plt.show()
