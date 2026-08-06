import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import sys
from datasets import Dataset, load_from_disk
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                    sys.stdout.flush()
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        sys.stdout.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.gpu = args.rank % torch.cuda.device_count()
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def load_and_filter(filter_data, nproc, input_data_file):
    data = load_from_disk(input_data_file)
    if filter_data is not None:
        data = filter_by_dict(data, filter_data, nproc)
    return data

def validate_and_clean_cols(train_data, eval_data, classifier):
    # validate that data has expected label column and remove others
    if classifier == "cell":
        label_col = "label"
    elif classifier == "gene":
        label_col = "labels"

    cols_to_keep = [label_col] + ["input_ids", "length"]
    if label_col not in train_data.column_names:
        print(f"train_data must contain column {label_col} with class labels.")
        raise
    else:
        train_data = remove_cols(train_data, cols_to_keep)

    if eval_data is not None:
        if label_col not in eval_data.column_names:
            print(
                f"eval_data must contain column {label_col} with class labels."
            )
            raise
        else:
            eval_data = remove_cols(eval_data, cols_to_keep)
    return train_data, eval_data

def remove_cols(data, cols_to_keep):
    other_cols = list(data.features.keys())
    other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
    data = data.remove_columns(other_cols)
    return data

def filter_by_dict(data, filter_data, nproc):
    for key, value in filter_data.items():

        def filter_data_by_criteria(example):
            return example[key] in value

        data = data.filter(filter_data_by_criteria, num_proc=nproc)
    if len(data) == 0:
        print("No cells remain after filtering. Check filtering criteria.")
        raise
    return data

def flatten_list(megalist):
    return [item for sublist in megalist for item in sublist]

def prep_gene_classifier_all_data(
    data, targets, labels, max_ncells, num_proc, balance=False
):
    targets = np.array(targets)
    labels = np.array(labels)
    label_dict_train = dict(zip(targets, labels))

    # function to filter by whether contains train labels
    def if_contains_train_label(example):
        a = targets
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    print("Filtering training data for genes to classify.")
    train_data = data.filter(if_contains_train_label, num_proc=num_proc)
    print(
        f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain\n"
    )

    if balance is True:
        train_data, label_dict_train = balance_gene_split(
            train_data, label_dict_train, num_proc
        )

    # subsample to max_ncells
    train_data = downsample_and_shuffle(train_data, max_ncells, None, None)

    # relabel genes for this split
    def train_classes_to_ids(example):
        example["labels"] = [
            label_dict_train.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)

    return train_data

def subsample_by_class(labels, N):
    label_indices = defaultdict(list)
    # Gather indices for each label
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    selected_indices = []
    # Select up to N indices for each label
    for label, indices in label_indices.items():
        if len(indices) > N:
            selected_indices.extend(random.sample(indices, N))
        else:
            selected_indices.extend(indices)
    return selected_indices

def downsample_and_shuffle(data, max_ncells, max_ncells_per_class, cell_state_dict):
    data = data.shuffle(seed=42)
    num_cells = len(data)
    # if max number of cells is defined, then subsample to this max number
    if max_ncells is not None:
        if num_cells > max_ncells:
            data = data.select([i for i in range(max_ncells)])
    if max_ncells_per_class is not None:
        class_labels = data[cell_state_dict["state_key"]]
        random.seed(42)
        subsample_indices = subsample_by_class(class_labels, max_ncells_per_class)
        data = data.select(subsample_indices)
    return data

def count_genes_for_balancing(subset_data, label_dict_subset, num_proc):
    def count_targets(example):
        labels = [
            label_dict_subset.get(token_id, -100) for token_id in example["input_ids"]
        ]
        counter_labels = Counter(labels)
        # get count of labels 0 or 1, or if absent, return 0
        example["labels_counts"] = [counter_labels.get(0, 0), counter_labels.get(1, 0)]
        return example

    subset_data = subset_data.map(count_targets, num_proc=num_proc)

    label0_counts = sum([counts[0] for counts in subset_data["labels_counts"]])
    label1_counts = sum([counts[1] for counts in subset_data["labels_counts"]])

    subset_data = subset_data.remove_columns("labels_counts")

    return label0_counts, label1_counts

def filter_data_balanced_genes(subset_data, label_dict_subset, num_proc):
    # function to filter by whether contains labels
    def if_contains_subset_label(example):
        a = list(label_dict_subset.keys())
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    print("Filtering data for balanced genes")
    subset_data_len_orig = len(subset_data)
    subset_data = subset_data.filter(if_contains_subset_label, num_proc=num_proc)
    print(
        f"Filtered {round((1-len(subset_data)/subset_data_len_orig)*100)}%; {len(subset_data)} remain\n"
    )

    return subset_data, label_dict_subset

def balance_gene_split(subset_data, label_dict_subset, num_proc):
    # count occurrence of genes in each label category
    label0_counts, label1_counts = count_genes_for_balancing(
        subset_data, label_dict_subset, num_proc
    )
    label_ratio_0to1 = label0_counts / label1_counts

    if 8 / 10 <= label_ratio_0to1 <= 10 / 8:
        # gene sets already balanced
        print(
            "Gene sets were already balanced within 0.8-1.25 fold and did not require balancing.\n"
        )
        return subset_data, label_dict_subset
    else:
        label_ratio_0to1_orig = label_ratio_0to1 + 0
        label_dict_subset_orig = label_dict_subset.copy()
        # balance gene sets
        max_ntrials = 25
        boost = 1
        if label_ratio_0to1 > 10 / 8:
            # downsample label 0
            for i in range(max_ntrials):
                label0 = 0
                label0_genes = [k for k, v in label_dict_subset.items() if v == label0]
                label0_ngenes = len(label0_genes)
                label0_nremove = max(
                    1,
                    int(
                        np.floor(
                            label0_ngenes - label0_ngenes / (label_ratio_0to1 * boost)
                        )
                    ),
                )
                random.seed(i)
                label0_remove_genes = random.sample(label0_genes, label0_nremove)
                label_dict_subset_new = {
                    k: v
                    for k, v in label_dict_subset.items()
                    if k not in label0_remove_genes
                }
                label0_counts, label1_counts = count_genes_for_balancing(
                    subset_data, label_dict_subset_new, num_proc
                )
                label_ratio_0to1 = label0_counts / label1_counts
                if 8 / 10 <= label_ratio_0to1 <= 10 / 8:
                    # if gene sets now balanced, return new filtered data and new label_dict_subset
                    return filter_data_balanced_genes(
                        subset_data, label_dict_subset_new, num_proc
                    )
                elif label_ratio_0to1 > 10 / 8:
                    boost = boost * 1.1
                elif label_ratio_0to1 < 8 / 10:
                    boost = boost * 0.9
        else:
            # downsample label 1
            for i in range(max_ntrials):
                label1 = 1
                label1_genes = [k for k, v in label_dict_subset.items() if v == label1]
                label1_ngenes = len(label1_genes)
                label1_nremove = max(
                    1,
                    int(
                        np.floor(
                            label1_ngenes
                            - label1_ngenes / ((1 / label_ratio_0to1) * boost)
                        )
                    ),
                )
                random.seed(i)
                label1_remove_genes = random.sample(label1_genes, label1_nremove)
                label_dict_subset_new = {
                    k: v
                    for k, v in label_dict_subset.items()
                    if k not in label1_remove_genes
                }
                label0_counts, label1_counts = count_genes_for_balancing(
                    subset_data, label_dict_subset_new, num_proc
                )
                label_ratio_0to1 = label0_counts / label1_counts
                if 8 / 10 <= label_ratio_0to1 <= 10 / 8:
                    # if gene sets now balanced, return new filtered data and new label_dict_subset
                    return filter_data_balanced_genes(
                        subset_data, label_dict_subset_new, num_proc
                    )
                elif label_ratio_0to1 < 8 / 10:
                    boost = boost * 1.1
                elif label_ratio_0to1 > 10 / 8:
                    boost = boost * 0.9

        assert i + 1 == max_ntrials
        if (label_ratio_0to1 <= label_ratio_0to1_orig < 8 / 10) or (
            10 / 8 > label_ratio_0to1_orig >= label_ratio_0to1
        ):
            label_ratio_0to1 = label_ratio_0to1_orig
            label_dict_subset_new = label_dict_subset_orig
        print(
            f"Gene sets were not able to be balanced within 0.8-1.25 fold after {max_ntrials} trials. Imbalance level: {label_ratio_0to1}\n"
        )
        return filter_data_balanced_genes(subset_data, label_dict_subset_new, num_proc)

def prep_gene_classifier_split(
    data,
    targets,
    labels,
    index,
    subset_name,
    max_ncells,
    num_proc,
    balance=False,
):
    # generate cross-validation splits
    targets = np.array(targets)
    labels = np.array(labels)
    targets_subset = targets[index]
    labels_subset = labels[index]
    label_dict_subset = dict(zip(targets_subset, labels_subset))

    # function to filter by whether contains train or eval labels
    def if_contains_subset_label(example):
        a = targets_subset
        b = example["input_ids"]
        return not set(a).isdisjoint(b)

    # filter dataset for examples containing classes for this split
    print(f"Filtering data for {subset_name} genes")
    subset_data = data.filter(if_contains_subset_label, num_proc=num_proc)
    print(
        f"Filtered {round((1-len(subset_data)/len(data))*100)}%; {len(subset_data)} remain\n"
    )

    # balance gene subsets if train
    if (subset_name == "train") and (balance is True):
        subset_data, label_dict_subset = balance_gene_split(
            subset_data, label_dict_subset, num_proc
        )

    # subsample to max_ncells
    subset_data = downsample_and_shuffle(subset_data, max_ncells, None, None)

    # relabel genes for this split
    def subset_classes_to_ids(example):
        example["labels"] = [
            label_dict_subset.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    subset_data = subset_data.map(subset_classes_to_ids, num_proc=num_proc)
    
    return subset_data

def prep_gene_classifier_train_eval_split(
    data,
    targets,
    labels,
    train_index,
    eval_index,
    max_ncells,
    num_proc,
    balance=False,
):
    # generate cross-validation splits
    train_data = prep_gene_classifier_split(
        data,
        targets,
        labels,
        train_index,
        "train",
        max_ncells,
        num_proc,
        balance,
    )
    eval_data = prep_gene_classifier_split(
        data,
        targets,
        labels,
        eval_index,
        "eval",
        max_ncells,
        num_proc,
        balance,
    )
    return train_data, eval_data

def remove_rare(data, rare_threshold, label, nproc=16):
    if rare_threshold > 0:
        total_cells = len(data)
        label_counter = Counter(data[label])
        nonrare_label_dict = {
            label: [k for k, v in label_counter if (v / total_cells) > rare_threshold]
        }
        data = filter_by_dict(data, nonrare_label_dict, nproc)
    return data

def label_gene_classes(example, class_id_dict, gene_class_dict):
    return [
        class_id_dict.get(gene_class_dict.get(token_id, -100), -100)
        for token_id in example["input_ids"]
    ]

def label_classes(classifier, data, gene_class_dict, nproc):
    if classifier == "cell":
        label_set = set(data["label"])
    elif classifier == "gene":
        # remove cells without any of the target genes
        def if_contains_label(example):
            a = flatten_list(gene_class_dict.values())
            b = example["input_ids"]
            return not set(a).isdisjoint(b)

        data = data.filter(if_contains_label, num_proc=nproc)
        label_set = gene_class_dict.keys()

        if len(data) == 0:
            print(
                "No cells remain after filtering for target genes. Check target gene list."
            )
            raise

    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    id_class_dict = {v: k for k, v in class_id_dict.items()}
    
    def classes_to_ids(example):
        if classifier == "cell":
            example["label"] = class_id_dict[example["label"]]
        elif classifier == "gene":
            example["labels"] = label_gene_classes(
                example, class_id_dict, gene_class_dict
            )
        return example

    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict

def vote(logit_list):
    m = max(logit_list)
    logit_list.index(m)
    indices = [i for i, x in enumerate(logit_list) if x == m]
    if len(indices) > 1:
        import random
        return random.choice(indices)
    else:
        return indices[0]

def get_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


# use for perturbation
def delete_indices(example):
    indices = example["perturb_index"]
    if any(isinstance(el, list) for el in indices):
        indices = flatten_list(indices)
    for index in sorted(indices, reverse=True):
        del example["input_ids"][index]

    example["length"] = len(example["input_ids"])
    return example

def overexpress_tokens(example, max_len, special_token):
    # -100 indicates tokens to overexpress are not present in rank value encoding
    if example["perturb_index"] != [-100]:
        example = delete_indices(example)
    if special_token:
        [
            example["input_ids"].insert(1, token)
            for token in example["tokens_to_perturb"][::-1]
        ]
    else:
        [
            example["input_ids"].insert(0, token)
            for token in example["tokens_to_perturb"][::-1]
        ]

    # truncate to max input size, must also truncate original emb to be comparable
    if len(example["input_ids"]) > max_len:
        if special_token:
            example["input_ids"] = example["input_ids"][0 : max_len - 1] + [
                example["input_ids"][-1]
            ]
        else:
            example["input_ids"] = example["input_ids"][0:max_len]
    example["length"] = len(example["input_ids"])
    return example
