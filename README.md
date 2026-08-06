# scAutoFM

Parameter-efficient **prompt supernet** fine-tuning and evolutionary architecture search on top of [Geneformer](https://huggingface.co/ctheodoris/Geneformer) for single-cell **cell-type / disease** classification and **gene property** prediction.

**Repository:** https://github.com/LaSiRy/scAutoFM

This public repository contains **code and experiment configs only**. Raw scRNA-seq data, processed HuggingFace datasets, checkpoints, figures, logs, and baseline method trees are **not** included (see `.gitignore`). Prepare data and pretrained Geneformer weights locally before training.

## Method (brief)

1. Train a **prompt supernet** over Geneformer (`supernet_train_prompt.py`).
2. Search a compact subnet with **evolution** / random / Bayesian search (`evolution.py`, etc.).
3. Retrain or evaluate the discovered subnet (`supernet_train_prompt.py --mode retrain`, `evaluation.py`).
4. Optional ablations: LoRA, Adapter, Prefix configs under `scAutoFM/experiments/`.

## Installation

```bash
git clone https://github.com/LaSiRy/scAutoFM.git
cd scAutoFM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install **Geneformer** and download a pretrained checkpoint (e.g. `gf-12L-30M-i2048`) from the official release:

- https://huggingface.co/ctheodoris/Geneformer

For baseline comparisons (not shipped here), see [docs/baselines.md](docs/baselines.md).

Working directory for the commands below is `scAutoFM/`:

```bash
cd scAutoFM
```

## Directory structure (tracked)

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── baselines.md
└── scAutoFM/
    ├── experiments/
    │   ├── scAutoFM/           # supernet + subnet YAML configs
    │   ├── LoRA/
    │   ├── Prefix/
    │   ├── P_Adapter/
    │   └── S_Adapter/
    ├── model/                  # supernet / MoE / adapter modules
    ├── lib/                    # datasets, collators, utils
    ├── supernet_train_prompt.py
    ├── supernet_engine_prompt.py
    ├── evolution.py
    ├── evaluation.py
    └── ...                     # search / plot helpers
```

Local-only (gitignored) directories you may create:

- `data/` — shared raw / tokenized corpora
- `scAutoFM/data/`, `scAutoFM/out/` — task datasets and splits
- `scAutoFM/saves/` — checkpoints
- `Geneformer/`, `scGPT/` — pretrained weights
- `baseline/` — private baseline runners and results

## Data

Datasets and checkpoints are **not** in this repo. Expected local layouts (examples):

- Cell tasks: `./out/<dataset>/{train,valid,test}/` with `.h5ad` or HuggingFace `.dataset` dirs
- Gene tasks: `./data/gene/<task_name>/` with k-fold labeled datasets

Point `--dataset_dir` / `--dataset_path` at your prepared paths.

## Train supernet

Cell-type example (aorta):

```bash
CUDA_VISIBLE_DEVICES=0 python supernet_train_prompt.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml" \
  --output_dir="./saves/supernet_aorta" \
  --batch-size=4 --lr=2e-4 --weight-decay=0.0001 \
  --dataset_dir="./out/aorta" \
  --label_name="cell_type" --nb_classes=12 \
  --custom-dataset --amp --epochs=50
```

Gene-task example:

```bash
CUDA_VISIBLE_DEVICES=0 python supernet_train_prompt.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml" \
  --output_dir="./saves/supernet_dosage_split1" \
  --lr=2e-4 --weight-decay=0.0001 \
  --task_type="gene" --amp --epochs=20 \
  --dataset_dir="./data/gene/dosage_sensitivity"
```

## Evolutionary search

```bash
CUDA_VISIBLE_DEVICES=0 python evolution.py \
  --cfg="./experiments/scAutoFM/supernet/supernet-A_prompt.yaml" \
  --output_dir="./saves/search_liver" \
  --resume="./saves/supernet_liver/checkpoint.pth" \
  --dataset_path="./out/liver" \
  --task_type="cell" --nb_classes=24 \
  --batch-size=32 --param-limits=3.0 \
  --max-epochs=20 --assessment="acc"
```

## Evaluate / retrain subnet

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --cfg="./experiments/scAutoFM/subnet/liver.yaml" \
  --resume="./saves/subnet_liver/checkpoint.pth" \
  --dataset_dir="./out/liver" \
  --task_type="cell" --custom-dataset --nb_classes=24 \
  --mode="retrain" --batch-size=128
```

Ablation configs (LoRA / Adapter / Prefix) live under `experiments/LoRA`, `experiments/P_Adapter`, `experiments/Prefix`, etc.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite Geneformer and the scAutoFM paper when available.
