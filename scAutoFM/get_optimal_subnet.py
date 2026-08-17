from lib import numpy_compat  # noqa: F401

"""Export the best searched architecture into a subnet YAML for retrain."""

import argparse
from pathlib import Path

import torch
import yaml


def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return (
        depth,
        list(cand_tuple[1:depth + 1]),
        list(cand_tuple[depth + 1: 2 * depth + 1]),
        list(cand_tuple[2 * depth + 1: 3 * depth + 1]),
        list(cand_tuple[3 * depth + 1: 4 * depth + 1]),
    )


def pick_best(vis_dict, keep_top_k, assessment):
    if 50 in keep_top_k and keep_top_k[50]:
        candidates = keep_top_k[50]
    else:
        # Fall back to the largest available top-k list.
        key = max(keep_top_k.keys())
        candidates = keep_top_k[key]
    if not candidates:
        raise RuntimeError("Search checkpoint has an empty keep_top_k list.")

    best = None
    best_score = None
    best_params = None
    for cand in candidates:
        info = vis_dict[cand]
        if assessment not in info:
            raise KeyError(
                f"Assessment '{assessment}' missing for candidate {cand}. "
                f"Available keys: {sorted(k for k in info if k != 'visited')}"
            )
        score = info[assessment]
        params = info.get("params", float("inf"))
        if (
            best is None
            or score > best_score
            or (score == best_score and params < best_params)
        ):
            best = cand
            best_score = score
            best_params = params
    return best, best_score, best_params


def load_template(template_cfg):
    if template_cfg is None:
        return {
            "MODEL_NAME": "geneformer",
            "SUPERNET": {
                "DEPTH": 12,
                "LORA_DIM": 100,
                "ADAPTER_DIM": 100,
                "PREFIX_DIM": 50,
            },
            "SEARCH_SPACE": {
                "LORA_DIM": [0, 5, 10, 50, 100],
                "ADAPTER_DIM": [0, 5, 10, 50, 100],
                "PREFIX_DIM": [0, 5, 10, 25, 50],
                "PREFIX_DEPTH": [3, 6, 9, 12],
                "LORA_DEPTH": [3, 6, 9, 12],
                "ADAPTER_DEPTH": [3, 6, 9, 12],
            },
        }
    with open(template_cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_subnet_cfg(template, lora_dim, s_adapter_dim, p_adapter_dim, prefix_dim):
    cfg = {
        "MODEL_NAME": template.get("MODEL_NAME", "geneformer"),
        "SUPERNET": dict(template.get("SUPERNET", {})),
        "SEARCH_SPACE": dict(template.get("SEARCH_SPACE", {})),
        "RETRAIN": {
            "LORA_DIM": list(lora_dim),
            "S_ADAPTER_DIM": list(s_adapter_dim),
            "P_ADAPTER_DIM": list(p_adapter_dim),
            "PREFIX_DIM": list(prefix_dim),
        },
    }
    # Ensure SUPERNET max dims cover the discovered architecture.
    supernet = cfg["SUPERNET"]
    supernet["DEPTH"] = max(int(supernet.get("DEPTH", len(lora_dim))), len(lora_dim))
    supernet["LORA_DIM"] = max(int(supernet.get("LORA_DIM", 0)), max(lora_dim) if lora_dim else 0)
    supernet["ADAPTER_DIM"] = max(
        int(supernet.get("ADAPTER_DIM", 0)),
        max(list(s_adapter_dim) + list(p_adapter_dim) + [0]),
    )
    supernet["PREFIX_DIM"] = max(
        int(supernet.get("PREFIX_DIM", 0)), max(prefix_dim) if prefix_dim else 0
    )
    return cfg


def get_args_parser():
    parser = argparse.ArgumentParser("Export optimal subnet YAML", add_help=False)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Search checkpoint produced by evolution.py (checkpoint-*.pth.tar)",
    )
    parser.add_argument(
        "--assessment",
        type=str,
        default="acc",
        help="Metric key stored in the search checkpoint (acc or auc)",
    )
    parser.add_argument(
        "--template_cfg",
        type=str,
        default="./experiments/scAutoFM/supernet/supernet-B_prompt.yaml",
        help="Base YAML whose SUPERNET/SEARCH_SPACE blocks are copied",
    )
    parser.add_argument(
        "--output_cfg",
        type=str,
        required=True,
        help="Path to write the subnet YAML used by --mode=retrain",
    )
    return parser


def main(args):
    info = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("vis_dict", "keep_top_k"):
        if key not in info:
            raise KeyError(f"Missing '{key}' in search checkpoint {args.checkpoint_path}")

    best, score, params = pick_best(info["vis_dict"], info["keep_top_k"], args.assessment)
    depth, lora_dim, s_adapter_dim, p_adapter_dim, prefix_dim = decode_cand_tuple(best)

    print(f"best_cand: {best}")
    print(f"assessment[{args.assessment}]: {score}")
    print(f"params(M): {params}")
    print(f"depth: {depth}")
    print(f"lora_dim: {lora_dim}")
    print(f"s_adapter_dim: {s_adapter_dim}")
    print(f"p_adapter_dim: {p_adapter_dim}")
    print(f"prefix_dim: {prefix_dim}")

    template = load_template(args.template_cfg)
    cfg = build_subnet_cfg(template, lora_dim, s_adapter_dim, p_adapter_dim, prefix_dim)

    out = Path(args.output_cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote subnet config: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "choose optimal subnet", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
