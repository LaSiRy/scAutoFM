# scAutoFM

Parameter-efficient **prompt supernet** fine-tuning and evolutionary architecture search on top of [Geneformer](https://huggingface.co/ctheodoris/Geneformer) for single-cell **cell-type / disease** classification and **gene property** prediction.

**Repository:** https://github.com/LaSiRy/scAutoFM

This repository ships **code + experiment YAML configs + Geneformer tokenizer sources**.  
Raw scRNA-seq data, HuggingFace `.dataset` dumps, checkpoints, figures, logs, and baseline trees are **not** included (see `.gitignore`). You must prepare pretrained weights and task datasets locally before training.

## Quick start (after `git clone`)

```bash
git clone https://github.com/LaSiRy/scAutoFM.git
cd scAutoFM

# 1) Python env (pick one)
## Option A — fresh venv
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
## Option B — existing conda env used in this project
#   conda env: hologenomnics
cd scAutoFM && source ./env_hologenomnics.sh

# 2) Always run training / search from the inner package directory
cd scAutoFM   # if not already there
```

All relative paths below (`../Geneformer/`, `./out/…`, `./experiments/…`) assume **cwd = `scAutoFM/`**.

## Environment & dependencies

| Item | Requirement |
|------|-------------|
| OS | Linux (CUDA GPU recommended) |
| Python | 3.10+ (tested with 3.10 in `hologenomnics`) |
| PyTorch | ≥ 2.0 with CUDA matching your driver |
| Key libs | `transformers` (<4.41), `datasets`, `timm`, `easydict`, `mmengine`, `anndata`, `scanpy`, `peft`, `pyyaml` |

Install from the repo root:

```bash
pip install -r requirements.txt
```

**hologenomnics tip:** `source ./env_hologenomnics.sh` sets `PYTHONNOUSERSITE=1` so `~/.local` packages do not break `timm` / `wandb` / NumPy. Entry scripts also load `lib/numpy_compat.py` as a NumPy 2.x shim for older wandb.

## Directory layout (after setup)

```
scAutoFM/                          # git clone root
├── README.md
├── LICENSE
├── requirements.txt
├── docs/baselines.md
├── Geneformer/                    # YOU create (pretrained weights; gitignored)
│   ├── gf-12L-30M-i2048/          # HF checkpoint (config.json + weights)
│   ├── geneformer/                # official package dicts (*.pkl)
│   └── token_dictionary_gc30M.pkl
└── scAutoFM/                      # code package (cwd for all commands)
    ├── Geneformer/                # tokenizer source (tracked)
    ├── experiments/scAutoFM/      # supernet + subnet YAMLs
    ├── experiments/{LoRA,Prefix,P_Adapter,S_Adapter}/
    ├── model/  lib/
    ├── out/<cell_dataset>/        # YOU create (cell tasks; gitignored)
    ├── data/gene/<gene_task>/     # YOU create (gene tasks; gitignored)
    └── saves/                     # checkpoints (gitignored)
```

## 1. Pretrained Geneformer weights

Download the **30M / 12-layer** Geneformer checkpoint used by this codebase (`gf-12L-30M-i2048`) from:

- https://huggingface.co/ctheodoris/Geneformer

Place it at the **repo root** (sibling of the inner `scAutoFM/` folder):

```text
Geneformer/gf-12L-30M-i2048/{config.json, pytorch_model.bin, ...}
Geneformer/geneformer/token_dictionary_gc30M.pkl
Geneformer/geneformer/gene_median_dictionary_gc30M.pkl
Geneformer/geneformer/gene_name_id_dict_gc30M.pkl
Geneformer/geneformer/ensembl_mapping_dict_gc30M.pkl
```

Training code loads weights via `../Geneformer/gf-12L-30M-i2048` and token dicts via `../Geneformer/geneformer/*.pkl`.  
The tracked package `scAutoFM/Geneformer/` resolves dictionary pickles from that same root install when local copies are absent.

Sanity check from `scAutoFM/`:

```bash
python -c "
from pathlib import Path
from Geneformer.tokenizer import TranscriptomeTokenizer
assert Path('../Geneformer/gf-12L-30M-i2048/config.json').exists()
assert Path('../Geneformer/geneformer/token_dictionary_gc30M.pkl').exists()
print('Geneformer OK', TranscriptomeTokenizer)
"
```

## 2. Datasets

Data are **not** in git. Prepare them under `scAutoFM/out/` (cell) or `scAutoFM/data/gene/` (gene).

### 2.1 Cell classification tasks

Expected layout for each dataset directory passed to `--dataset_dir` / `--dataset_path`:

```text
out/<name>/
  train.dataset/     # HuggingFace datasets.save_to_disk
  valid.dataset/
  [test.dataset/]    # optional; used by some search scripts
  *_id_class_dict.pkl
```

Each split must contain at least columns: **`input_ids`**, **`label`**, **`length`**.

| Dataset dir | Task | Label field (when preparing) | `--nb_classes` | Example sizes (train / valid) |
|-------------|------|------------------------------|----------------|-------------------------------|
| `./out/aorta` | cell type | `cell_type` | **12** | ~7700 / ~962 |
| `./out/liver` | cell type | `cell_type` | **24** | ~17771 / ~2221 |
| `./out/kidney` | cell type | `cell_type` | **13** | ~9100 / ~1138 |
| `./out/kidney_balance` | cell type (balanced) | `cell_type` | **13** | ~2704 / ~338 |
| `./out/human_dcm_hcm` | disease | `disease` | **3** (NF / DCM / HCM) | ~104k / ~23k |

**Prepare from `.h5ad`** (Ensembl IDs required):

1. Ensure `adata.var` has `ensembl_id` (or `gene_ids`) and cells have raw counts; set `adata.obs["n_counts"]`.
2. Use `prepare_data.py` / Geneformer `TranscriptomeTokenizer` to write tokenized HuggingFace datasets, then rename/split to `train.dataset` + `valid.dataset` under `./out/<name>/`.
3. Keep an `*_id_class_dict.pkl` mapping `{id: class_name}`; set `--nb_classes` to `len(dict)`.

Minimal check:

```bash
python -c "
from datasets import load_from_disk
d = load_from_disk('./out/aorta/train.dataset')
print(len(d), d.features)
"
```

### 2.2 Gene classification tasks (5-fold)

Expected layout for `--dataset_dir` / `--dataset_path`:

```text
data/gene/<task>/
  train_gene_labeled_ksplit{1..5}.dataset/
  valid_gene_labeled_ksplit{1..5}.dataset/
  test_gene_labeled_ksplit{1..5}.dataset/
```

| Task dir | Description | Classes | Select fold |
|----------|-------------|---------|-------------|
| `./data/gene/dosage_sensitivity` | Dosage-sensitive TF classification | 2 | `--dataset_id 1` … `5` |
| `./data/gene/bivalent_vs_lys4_only` | Bivalent vs H3K4me1-only | 2 | `--dataset_id …` |
| `./data/gene/bivalent_vs_no_methyl` | Bivalent vs no methylation | 2 | `--dataset_id …` |

Gene loaders use `*_ksplit{dataset_id}.dataset` (default `--dataset_id 1`).  
Binary gene tasks use `--task_type gene` (model sets `nb_classes=2` for token classification).

These labeled folds typically come from the Geneformer gene-classifier preprocessing pipeline (see official Geneformer docs / classifier utilities). Place the resulting `.dataset` folders under `data/gene/<task>/` as above.

## 3. Configs

| Path | Role |
|------|------|
| `experiments/scAutoFM/supernet/supernet-B_prompt.yaml` | Default prompt supernet search space (cell + gene) |
| `experiments/scAutoFM/supernet/supernet-A_prompt.yaml` | Alternate supernet space (used in some search runs) |
| `experiments/scAutoFM/subnet/<dataset>.yaml` | Fixed subnet for `--mode retrain` (contains `RETRAIN:` dims) |
| `experiments/LoRA|Prefix|P_Adapter|S_Adapter/*.yaml` | Ablations |

YAML keys: `SUPERNET` (max dims / depth), `SEARCH_SPACE` (candidate dims), optional `RETRAIN` (per-layer dims for subnet training).

## 4. Pipeline (supernet → search → subnet)

Working directory: **`scAutoFM/`**.

### 4.1 Train prompt supernet

**Cell (aorta example):**

```bash
CUDA_VISIBLE_DEVICES=0 python supernet_train.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml" \
  --mode=super \
  --task_type=cell \
  --custom-dataset \
  --dataset_dir="./out/aorta" \
  --label_name=cell_type \
  --nb_classes=12 \
  --batch-size=4 --lr=2e-4 --weight-decay=0.0001 \
  --epochs=50 --amp \
  --output_dir="./saves/supernet_aorta"
```

Other cell datasets (change dir / classes / label):

```bash
# liver — 24 cell types
--dataset_dir=./out/liver --nb_classes=24 --label_name=cell_type --output_dir=./saves/supernet_liver

# kidney_balance — 13 cell types
--dataset_dir=./out/kidney_balance --nb_classes=13 --label_name=cell_type --output_dir=./saves/supernet_kidney

# human DCM/HCM — 3 disease classes
--dataset_dir=./out/human_dcm_hcm --nb_classes=3 --label_name=disease --output_dir=./saves/supernet_human
```

**Gene (dosage sensitivity, fold 1):**

```bash
CUDA_VISIBLE_DEVICES=0 python supernet_train.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml" \
  --mode=super \
  --task_type=gene \
  --dataset_dir="./data/gene/dosage_sensitivity" \
  --dataset_id=1 \
  --lr=4e-4 --epochs=20 --amp \
  --output_dir="./saves/supernet_dosage_sensitivity_split1"
```

Checkpoint: `./saves/<run>/checkpoint.pth`.

### 4.2 Evolutionary search

Cell search loads **`valid.dataset`** from `--dataset_path` and needs a trained supernet `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0 python evolution.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-A_prompt.yaml" \
  --mode=search \
  --task_type=cell \
  --dataset_path="./out/liver" \
  --nb_classes=24 \
  --resume="./saves/supernet_liver/checkpoint.pth" \
  --batch-size=24 --param-limits=5.0 \
  --max-epochs=20 --assessment=acc \
  --output_dir="./saves/search_liver"
```

Gene search example:

```bash
CUDA_VISIBLE_DEVICES=0 python evolution.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml" \
  --mode=search \
  --task_type=gene \
  --dataset_path="./data/gene/dosage_sensitivity" \
  --dataset_id=1 \
  --resume="./saves/supernet_dosage_sensitivity_split1/checkpoint.pth" \
  --assessment=auc \
  --output_dir="./saves/search_dosage_sensitivity_split1"
```

Copy the discovered architecture into a subnet YAML under `experiments/scAutoFM/subnet/` (`RETRAIN:` block), or use an existing one (e.g. `aorta.yaml`, `liver.yaml`, `dosage.yaml`).

### 4.3 Retrain / evaluate subnet

```bash
# retrain discovered aorta subnet
CUDA_VISIBLE_DEVICES=0 python supernet_train.py \
  --cfg="./experiments/scAutoFM/subnet/aorta.yaml" \
  --mode=retrain \
  --task_type=cell \
  --custom-dataset \
  --dataset_dir="./out/aorta" \
  --nb_classes=12 \
  --batch-size=4 --lr=2e-4 --epochs=50 --amp \
  --output_dir="./saves/subnet_aorta"

# evaluate
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --cfg="./experiments/scAutoFM/subnet/aorta.yaml" \
  --mode=retrain \
  --task_type=cell \
  --custom-dataset \
  --dataset_dir="./out/aorta" \
  --nb_classes=12 \
  --resume="./saves/subnet_aorta/checkpoint.pth" \
  --eval --amp
```

Gene subnet example:

```bash
CUDA_VISIBLE_DEVICES=0 python supernet_train.py \
  --cfg="./experiments/scAutoFM/subnet/dosage.yaml" \
  --mode=retrain \
  --task_type=gene \
  --dataset_dir="./data/gene/dosage_sensitivity" \
  --dataset_id=1 \
  --epochs=20 --amp \
  --output_dir="./saves/subnet_dosage_split1"
```

## 5. Checklist: “pull → runnable”

1. [ ] Clone repo; `cd scAutoFM` (inner) for all commands  
2. [ ] Install `requirements.txt` (or `source ./env_hologenomnics.sh`)  
3. [ ] Place Geneformer `gf-12L-30M-i2048` + `geneformer/*.pkl` under repo-root `Geneformer/`  
4. [ ] Prepare at least one cell dataset under `out/<name>/{train,valid}.dataset` **or** a gene fold under `data/gene/<task>/`  
5. [ ] Run the Geneformer sanity `python -c …` above  
6. [ ] Train supernet → search → retrain with matching `--nb_classes` / `--dataset_id`  

If imports fail with NumPy / wandb errors, use `PYTHONNOUSERSITE=1` or `env_hologenomnics.sh`.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Please cite Geneformer and the scAutoFM paper when available. Baseline method pointers: [docs/baselines.md](docs/baselines.md).
