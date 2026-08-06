from pathlib import Path

_HERE = Path(__file__).parent
# Prefer local copies; fall back to repo-root Geneformer install (../Geneformer/geneformer/).
_ROOT_GENEFORMER = _HERE.parent.parent / "Geneformer" / "geneformer"


def _resolve(name: str) -> Path:
    local = _HERE / name
    if local.exists():
        return local
    alt = _ROOT_GENEFORMER / name
    return alt if alt.exists() else local


GENE_MEDIAN_FILE = _resolve("gene_median_dictionary_gc30M.pkl")
TOKEN_DICTIONARY_FILE = _resolve("token_dictionary_gc30M.pkl")
ENSEMBL_DICTIONARY_FILE = _resolve("gene_name_id_dict_gc30M.pkl")
ENSEMBL_MAPPING_FILE = _resolve("ensembl_mapping_dict_gc30M.pkl")

from . import tokenizer
from .tokenizer import TranscriptomeTokenizer
