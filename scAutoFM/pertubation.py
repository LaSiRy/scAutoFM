from lib import numpy_compat  # noqa: F401
from lib.in_silico_perturber import InSilicoPerturber
from lib.in_silico_perturber_stats import InSilicoPerturberStats
from lib.emb_extractor import EmbExtractor
from lib.config import cfg, update_config_from_file
import torch
import argparse
# get model
from model.supernet import GeneFormer
from transformers import AutoTokenizer, BertForSequenceClassification, BertForTokenClassification, BertConfig, get_cosine_schedule_with_warmup

def perturbation(args):
    config = BertConfig.from_pretrained('../Geneformer/gf-12L-30M-i2048/config.json')
    update_config_from_file(args.cfg)
    basemodel = BertForSequenceClassification.from_pretrained('../Geneformer/gf-12L-30M-i2048')
    model = GeneFormer(
        config=config,
        basemodel=basemodel, 
        num_classes=args.nb_classes, 
        pool=False if args.task_type == "gene" else True,
        weight_init='',
        LoRA_dim=cfg.SUPERNET.LORA_DIM,
        adapter_dim=cfg.SUPERNET.ADAPTER_DIM,
        prefix_dim=cfg.SUPERNET.PREFIX_DIM,
        drop_rate_LoRA=args.drop_rate_LoRA,
        drop_rate_adapter=args.drop_rate_adapter
        )

    choices = {'depth': cfg.SUPERNET.DEPTH,
        'super_LoRA_dim':cfg.SUPERNET.LORA_DIM,
        'super_adapter_dim':cfg.SUPERNET.ADAPTER_DIM,
        'super_prefix_dim':cfg.SUPERNET.PREFIX_DIM,
        'lora_dim':cfg.SEARCH_SPACE.LORA_DIM,
        'adapter_dim':cfg.SEARCH_SPACE.ADAPTER_DIM,
        'prefix_dim':cfg.SEARCH_SPACE.PREFIX_DIM,
        'lora_depth':cfg.SEARCH_SPACE.LORA_DEPTH,
        'adapter_depth':cfg.SEARCH_SPACE.ADAPTER_DEPTH,
        'prefix_depth':cfg.SEARCH_SPACE.PREFIX_DEPTH,
        }


    device = torch.device(args.device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False) #
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    config = {'lora_dim':cfg.RETRAIN.LORA_DIM,'p_adapter_dim':cfg.RETRAIN.P_ADAPTER_DIM,'s_adapter_dim':cfg.RETRAIN.S_ADAPTER_DIM,'prefix_dim':cfg.RETRAIN.PREFIX_DIM,}
    model.set_sample_config(config=config)

    # first obtain start, goal, and alt embedding positions
    # this function was changed to be separate from perturb_data
    # to avoid repeating calcuations when parallelizing perturb_data
    cell_states_to_model={"state_key": "disease", 
                        "start_state": "NF", 
                        "goal_state": "DCM", 
                        "alt_states": ["HCM"]}

    # OF NOTE: token_dictionary_file must be set to the gc-30M token dictionary if using a 30M series model
    # (otherwise the EmbExtractor will use the current default model dictionary)
    # 30M token dictionary: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
    # embex = EmbExtractor(model_type="CellClassifier",
    #                     num_classes=3,
    #                     filter_data=None,
    #                     emb_mode="cell",
    #                     max_ncells=500,
    #                     emb_layer=0,
    #                     summary_stat="exact_mean",
    #                     forward_batch_size=50,
    #                     nproc=8)

    # state_embs_dict = embex.get_state_embs(model,
    #                                     cell_states_to_model,
    #                                     './out/human_dcm_hcm/human_dcm_hcm.dataset',
    #                                     './out/perturbation_fromNF',
    #                                     "perturbation")

    # # OF NOTE: token_dictionary_file must be set to the gc-30M token dictionary if using a 30M series model
    # # (otherwise the InSilicoPerturber will use the current default model dictionary)
    # # 30M token dictionary: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
    # isp = InSilicoPerturber(perturb_type="delete",
    #                         perturb_rank_shift=None,
    #                         genes_to_perturb="all",
    #                         combos=0,
    #                         anchor_gene=None,
    #                         model_type="CellClassifier", # if using previously fine-tuned cell classifier model
    #                         num_classes=3,
    #                         emb_mode="cell",
    #                         cell_emb_style="mean_pool",
    #                         filter_data=None,
    #                         cell_states_to_model=cell_states_to_model,
    #                         state_embs_dict=state_embs_dict,
    #                         max_ncells=500,
    #                         emb_layer=0,
    #                         forward_batch_size=50,
    #                         nproc=10)

    # # outputs intermediate files from in silico perturbation

    # isp.perturb_data(model,
    #                 './out/human_dcm_hcm/human_dcm_hcm.dataset',
    #                 './out/perturbation_fromNF',
    #                 "perturb_human")

    # OF NOTE: token_dictionary_file must be set to the gc-30M token dictionary if using a 30M series model
    # (otherwise the InSilicoPerturberStats will use the current default model dictionary)
    # 30M token dictionary: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
    gene_list=["ENSG00000078814","ENSG00000155657","ENSG00000160789","ENSG00000134571","ENSG00000136574","ENSG00000159251","ENSG00000198523","ENSG00000183873","ENSG00000151150","ENSG00000197580","ENSG00000140479", "ENSG00000089225", "ENSG00000169946", "ENSG00000183072","ENSG00000175445"]
    ispstats = InSilicoPerturberStats(mode="aggregate_data",
                                    genes_perturbed=gene_list,
                                    combos=0,
                                    anchor_gene=None,
                                    cell_states_to_model=cell_states_to_model)

    # extracts data from intermediate files and processes stats to output in final .csv
    ispstats.get_stats('./out/perturbation_fromNF', # this should be the directory 
                    None,
                    './out/out_perturb_stat',
                    "perturbation_fromNF_embedding")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perturbation', add_help=False)

    # config file
    parser.add_argument('--cfg',help='experiment configure file name',default='./experiments/scAutoFM/subnet/human.yaml', type=str)

    # Model parameters
    parser.add_argument('--model', default='geneformer', type=str,
                        help='Name of model to train')
    parser.add_argument('--task_type', default='cell', type=str,
                        help='task_type: cell or gene')
    
    parser.add_argument('--nb_classes', default=3, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='./saves/subnet_human/checkpoint.pth', help='resume from checkpoint')

    parser.add_argument('--label_name', default="cell_type", type=str)

    parser.add_argument('--drop_rate_LoRA', type=float, default=0.1)
    parser.add_argument('--drop_rate_prompt', type=float, default=0.1)
    parser.add_argument('--drop_rate_adapter', type=float, default=0.1)
    
    args = parser.parse_args()
    perturbation(args)