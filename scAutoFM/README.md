# scAutoFM

Training, evolutionary search, and evaluation entry points for **scAutoFM** (prompt supernet on Geneformer).

**Start here:** [repository root README](../README.md) — environment, Geneformer weights, dataset layouts, and full train → search → retrain commands.

### Local (hologenomnics)

```bash
source ./env_hologenomnics.sh
# cwd must stay this directory
```

### Configs

- Supernet: `experiments/scAutoFM/supernet/`
- Subnet / retrain: `experiments/scAutoFM/subnet/`
- Ablations: `experiments/LoRA`, `Prefix`, `P_Adapter`, `S_Adapter`
