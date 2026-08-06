import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
# 假设有多个模型的预测概率和真实标签
# "bivalent_vs_non_methylated", "bivalent_vs_lys4_only_methylated", "dosage_sensitive_tf", "long_short_range_tf"
def get_aucs():
    paths = ['../baseline/scPEFTNAS/range_1_results.pkl',
             '../baseline/scPEFTNAS/range_2_results.pkl',
             '../baseline/scPEFTNAS/range_3_results.pkl',
             '../baseline/scPEFTNAS/range_4_results.pkl',
             '../baseline/scPEFTNAS/range_5_results.pkl',
             ]
    tpr_list = []
    roc_auc = []
    fpr_list = []
    import numpy as np
    for path in paths:
        with open(path, "rb") as f:
            metric = pickle.load(f)
        fpr_list.append(metric['fpr'])
        tpr_list.append(metric['tpr'])
        roc_auc.append(metric['auc'])
    mean_fpr = np.linspace(0, 1, 100)
    tpr_interp = np.array([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)])
    mean_tpr = np.mean(tpr_interp, axis=0)
    avg_roc_auc = np.mean(roc_auc)
    roc_auc_sd = np.std(roc_auc)
    metrics = {
        'all_roc_metrics': {
            'mean_tpr':mean_tpr,
            'mean_fpr':mean_fpr,
            'all_roc_auc': roc_auc,
            'roc_auc': avg_roc_auc,
            'roc_auc_sd': roc_auc_sd,
        }
    }
    with open(f'../baseline/scPEFTNAS/long_short_range_tf.pkl', "wb") as f:
        pickle.dump(metrics, f)
    print(metrics)

def get_models_auc(task_name):
    # get gene2vec-rf & -lr
    with open(f'../baseline/gene2vec/rf/{task_name}.pkl', "rb") as f:
        metric_gene2vec_rf = pickle.load(f)
    with open(f'../baseline/gene2vec/lr/{task_name}.pkl', "rb") as f:
        metric_gene2vec_lr = pickle.load(f)
    # get genePT-rf & -lr
    with open(f'../baseline/genePT/rf/{task_name}.pkl', "rb") as f:
        metric_genept_rf = pickle.load(f)
    with open(f'../baseline/genePT/lr/{task_name}.pkl', "rb") as f:
        metric_genept_lr = pickle.load(f)
    # get geneformer
    with open(f'../baseline/geneformer/{task_name}.pkl', "rb") as f:
        metric_geneformer = pickle.load(f)

    # get mine
    with open(f'../baseline/scPEFTNAS/{task_name}.pkl', "rb") as f:
        metric_scPEFTNAS = pickle.load(f)
    metrics = {
        'Gene2vec-rf': metric_gene2vec_rf['all_roc_metrics'],
        'Gene2vec-lr': metric_gene2vec_lr['all_roc_metrics'],
        'GenePT-rf': metric_genept_rf['all_roc_metrics'],
        'GenePT-lr': metric_genept_lr['all_roc_metrics'],
        'Geneformer': metric_geneformer['all_roc_metrics'],
        'scPEFTNAS': metric_scPEFTNAS['all_roc_metrics'],
    }
    return metrics

tasks = ["bivalent_vs_non_methylated", "bivalent_vs_lys4_only_methylated", "long_short_range_tf"]
model_style_dict = {'Gene2vec-rf': {'color': 'grey', 'linestyle': '-'},
        'Gene2vec-lr': {'color': 'orange', 'linestyle': '-'},
        'GenePT-rf': {'color': 'cyan', 'linestyle': '-'},
        'GenePT-lr': {'color': 'blue', 'linestyle': '-'},
        'Geneformer':{'color': 'purple', 'linestyle': '-'},
        'scPEFTNAS': {'color': 'red', 'linestyle': '-'},
        }
for task in tasks:
    plt.figure(figsize=(10, 8))
    lw = 3
    metrics = get_models_auc(task)
    for model_name in metrics.keys():
        mean_fpr = metrics[model_name]["mean_fpr"]
        mean_tpr = metrics[model_name]["mean_tpr"]
        roc_auc = metrics[model_name]["roc_auc"]
        roc_auc_sd = metrics[model_name]["roc_auc_sd"]
        color = model_style_dict[model_name]["color"]
        linestyle = model_style_dict[model_name]["linestyle"]
        if len(metrics[model_name]["all_roc_auc"]) > 1:
            label = f"{model_name} (AUC {roc_auc:0.2f} $\pm$ {roc_auc_sd:0.2f})"
        else:
            label = f"{model_name} (AUC {roc_auc:0.2f})"
        plt.plot(
            mean_fpr, mean_tpr, color=color, linestyle=linestyle, lw=lw, label=label
        )
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f"{task}_auc.png", dpi=300, bbox_inches='tight')
