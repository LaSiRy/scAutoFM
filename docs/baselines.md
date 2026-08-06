# Baselines

This repository ships **scAutoFM** training and evaluation code only.
Baseline method code, pretrained weights, and experiment outputs are **not**
included in the public GitHub tree (see root `.gitignore`).

Comparisons in the paper/experiments used the following external projects.
Install and run them from their official sources:

| Method | Official repository / resource |
|--------|--------------------------------|
| Geneformer | [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) |
| scGPT | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) |
| GeneCompass | [xLifeGroup/GeneCompass](https://github.com/xLifeGroup/GeneCompass) (or the authors’ published release) |

Local development may keep baseline runners and results under a private
`baseline/` directory; that path is gitignored and must not be pushed.
