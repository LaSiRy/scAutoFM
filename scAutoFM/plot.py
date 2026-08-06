import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, save_path, labels=None):
    # 计算总和用于归一化
    true_0 = np.sum(cm[0])
    true_1 = np.sum(cm[1])
    cm_normalized = [[cm[0][0]/true_0, cm[0][1]/true_0], [cm[1][0]/true_1, cm[1][1]/true_1]]

    # 设置类别标签
    true_labels = [f'{labels[0]}\nn={true_0}', f'{labels[1]}\nn={true_1}']
    pred_labels = [f'{labels[0]}\nn={true_0}', f'{labels[1]}\nn={true_1}']

    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=pred_labels, yticklabels=true_labels)

    # 设置标题和轴标签
    plt.title('scNAS', fontsize=14, pad=20)
    plt.xlabel('Predicted label', fontsize=12, labelpad=15)
    plt.ylabel('True label', fontsize=12, labelpad=15)

    # 调整颜色条位置和标签
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    # 调整文本颜色以便于阅读
    for text in plt.gca().texts:
        if float(text.get_text()) < 0.5:
            text.set_color('black')
        else:
            text.set_color('white')

    # 显示图形
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_roc_auc(fpr, tpr, roc_auc, roc_auc_sd, cross_valid, save_path, labels=None):
    if cross_valid:
        mean_fpr=np.linspace(0,1,100)
        interp_tprs = []

        for _fpr, _tpr in zip(fpr, tpr):
            interp_tpr = np.interp(mean_fpr, _fpr, _tpr)
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        plt.figure(figsize=(8, 7))
        plt.plot(mean_fpr, mean_tpr, color='red', lw=2, label=f'scNAS (AUC {roc_auc:.3f} ± {roc_auc_sd:.3f})')
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # 对角线

    else:
        plt.figure(figsize=(8, 7))
        plt.plot(fpr, tpr, color='red', lw=2, label=f'scNAS (AUC {roc_auc:.2f} ± {roc_auc_sd:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    # 设置标题和轴标签
    plt.title(labels, fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    # 设置图例
    plt.legend(loc='lower right', fontsize=10)

    # 设置网格和边界
    plt.grid(False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # 显示图形
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

