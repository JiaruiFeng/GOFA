import torch

from tasks import GOFAPretrainTaskWrapper
import gc
import random
import numpy as np

DATA_ROOT = "/storage1/yinjie.tang/Active/feng.jiarui/TAGDataset"
SAVE_NAME_BASE = "pretrain"
CS_MAX_LEFT_KEEP_LENGTH = 128

def generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences,num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name, num_LP=1, SP_from_targets=True, CN_from_targets=True):

    node_task = GOFAPretrainTaskWrapper(task_names=dataset,
                                        root=DATA_ROOT,
                                        split=split,
                                        sample_size=sample_range,
                                        save_name=node_task_save_name,
                                        save_data=True,
                                        from_saved=False,
                                        num_workers=num_workers,
                                        hop=hop,
                                        max_nodes_per_hop=num_nodes_per_hop,
                                        pretrain_tasks=node_task_list,
                                        num_additional_sentences=additional_sentences,
                                        num_SP=num_SP,
                                        num_CN=num_CN,
                                        include_targets=include_targets,
                                        left_keep_length=CS_MAX_LEFT_KEEP_LENGTH,
                                        num_LP=num_LP,
                                        SP_from_targets=SP_from_targets,
                                        CN_from_targets=CN_from_targets,
                                        )
    del node_task
    gc.collect()
    key_to_content_task = GOFAPretrainTaskWrapper(task_names=dataset,
                                                  root=DATA_ROOT,
                                                  split=split,
                                                  sample_size=key_to_content_sample_range,
                                                  save_name=key_to_content_task_save_name,
                                                  save_data=True,
                                                  from_saved=False,
                                                  num_workers=num_workers,
                                                  hop=hop,
                                                  max_nodes_per_hop=num_nodes_per_hop,
                                                  pretrain_tasks=["IR"]
                                                 )
    del key_to_content_task
    gc.collect()
    content_to_key_task = GOFAPretrainTaskWrapper(task_names=dataset,
                                                  root=DATA_ROOT,
                                                  split=split,
                                                  sample_size=content_to_key_sample_range,
                                                  save_name=content_to_key_task_save_name,
                                                  save_data=True,
                                                  from_saved=False,
                                                  num_workers=num_workers,
                                                  hop=hop,
                                                  max_nodes_per_hop=num_nodes_per_hop,
                                                  pretrain_tasks=["IR"],
                                                  content_to_key=True
                                                  )
    del content_to_key_task
    gc.collect()

def generate_mag240m(epoch):
    dataset = "mag240m"
    node_task_list = ["CS", "CN", "SP"]
    node_task_sample_size_per_epoch = 500_000
    IR_task_sample_size_per_epoch = 10_000
    sample_range = [[epoch * node_task_sample_size_per_epoch + i for i in range(node_task_sample_size_per_epoch)]]
    key_to_content_sample_range = [[4_500_000 + epoch * IR_task_sample_size_per_epoch + i for i in range(IR_task_sample_size_per_epoch)]]
    content_to_key_sample_range = [[5_000_000 + epoch * IR_task_sample_size_per_epoch + i for i in range(IR_task_sample_size_per_epoch)]]
    additional_sentences = 3
    include_targets = True
    num_SP = 3
    num_CN = 3
    hop = 3
    num_nodes_per_hop = 5
    num_workers = 32
    split = "all"
    node_task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    key_to_content_task_save_name = "_".join([SAVE_NAME_BASE, "IR_kc", str(epoch)])
    content_to_key_task_save_name = "_".join([SAVE_NAME_BASE, "IR_ck", str(epoch)])
    generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences,num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name)

def generate_arxiv(epoch):
    dataset = "arxiv"
    node_task_list = ["CS", "CN", "SP"]
    node_task_sample_size_per_epoch = 50_000
    IR_task_sample_size_per_epoch = 10_000
    sample_range = [[epoch * node_task_sample_size_per_epoch + i for i in range(node_task_sample_size_per_epoch)]]

    key_to_content_sample_range = [i for i in range(169_343)]
    for _ in range(epoch + 1):
        random.shuffle(key_to_content_sample_range)
    key_to_content_sample_range = [key_to_content_sample_range[:IR_task_sample_size_per_epoch]]

    content_to_key_sample_range = [i for i in range(169_343)]
    for _ in range(epoch + 1):
        random.shuffle(content_to_key_sample_range)
    content_to_key_sample_range = [content_to_key_sample_range[:IR_task_sample_size_per_epoch]]

    additional_sentences = 3
    include_targets = True
    num_SP = 3
    num_CN = 3
    hop = 3
    num_nodes_per_hop = 5
    num_workers = 32
    split = "all"
    node_task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    key_to_content_task_save_name = "_".join([SAVE_NAME_BASE, "IR_kc", str(epoch)])
    content_to_key_task_save_name = "_".join([SAVE_NAME_BASE, "IR_ck", str(epoch)])
    generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences, num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name)

def generate_pubmed_node(epoch):
    dataset = "pubmed_node"
    node_task_list = ["CS", "CN", "SP"]
    node_task_sample_size_per_epoch = 5_000
    IR_task_sample_size_per_epoch = 5_000
    sample_range = [[epoch * node_task_sample_size_per_epoch + i for i in range(node_task_sample_size_per_epoch)]]
    key_to_content_sample_range = [i for i in range(19_717)]
    for _ in range(epoch + 1):
        random.shuffle(key_to_content_sample_range)
    key_to_content_sample_range = [key_to_content_sample_range[:IR_task_sample_size_per_epoch]]

    content_to_key_sample_range = [i for i in range(19_717)]
    for _ in range(epoch + 1):
        random.shuffle(content_to_key_sample_range)
    content_to_key_sample_range = [content_to_key_sample_range[:IR_task_sample_size_per_epoch]]
    additional_sentences = 3
    include_targets = True
    num_SP = 3
    num_CN = 3
    hop = 3
    num_nodes_per_hop = 5
    num_workers = 32
    split = "all"
    node_task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    key_to_content_task_save_name = "_".join([SAVE_NAME_BASE, "IR_kc", str(epoch)])
    content_to_key_task_save_name = "_".join([SAVE_NAME_BASE, "IR_ck", str(epoch)])
    generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences, num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name)


def generate_ultrachat200k(epoch):
    dataset = "ultrachat200k"
    task_list = ["DS"]
    task_sample_size_per_epoch = 100_000
    sample_range = [[epoch * task_sample_size_per_epoch + i for i in range(task_sample_size_per_epoch)]]
    split = "all"
    num_workers = 32
    task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    task = GOFAPretrainTaskWrapper(task_names=dataset,
                                        root=DATA_ROOT,
                                        split=split,
                                        sample_size=sample_range,
                                        save_name=task_save_name,
                                        save_data=True,
                                        from_saved=False,
                                        num_workers=num_workers,
                                        pretrain_tasks=task_list,
                                        )
    del task
    gc.collect()

def generate_wiki_graph(epoch):
    dataset = "wiki_graph"
    node_task_list = ["CS", "CN", "SP", "LP"]
    node_task_sample_size_per_epoch = 80_000
    IR_task_sample_size_per_epoch = 10_000
    sample_range = [[epoch * node_task_sample_size_per_epoch + i for i in range(node_task_sample_size_per_epoch)]]

    key_to_content_sample_range = [i for i in range(240_000)]
    for _ in range(epoch + 1):
        random.shuffle(key_to_content_sample_range)
    key_to_content_sample_range = [key_to_content_sample_range[:IR_task_sample_size_per_epoch]]

    content_to_key_sample_range = [i for i in range(240_000)]
    for _ in range(epoch + 1):
        random.shuffle(content_to_key_sample_range)
    content_to_key_sample_range = [content_to_key_sample_range[:IR_task_sample_size_per_epoch]]

    additional_sentences = 4
    include_targets = False
    SP_from_targets = False
    CN_from_targets = False
    num_SP = 2
    num_CN = 2
    num_LP = 2
    hop = 3
    num_nodes_per_hop = 5
    num_workers = 32
    split = "train"
    node_task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    key_to_content_task_save_name = "_".join([SAVE_NAME_BASE, "IR_kc", str(epoch)])
    content_to_key_task_save_name = "_".join([SAVE_NAME_BASE, "IR_ck", str(epoch)])
    generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences, num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name, num_LP, SP_from_targets, CN_from_targets)
def generate_wikikg90m(epoch):
    dataset = "wikikg90m"
    node_task_list = ["CS", "CN", "SP", "LP"]
    node_task_sample_size_per_epoch = 100_000
    IR_task_sample_size_per_epoch = 10_000
    sample_range = [[epoch * node_task_sample_size_per_epoch + i for i in range(node_task_sample_size_per_epoch)]]

    key_to_content_sample_range = [i for i in range(100_000_000)]
    for _ in range(epoch + 1):
        random.shuffle(key_to_content_sample_range)
    key_to_content_sample_range = [key_to_content_sample_range[:IR_task_sample_size_per_epoch]]

    content_to_key_sample_range = [i for i in range(100_000_000)]
    for _ in range(epoch + 1):
        random.shuffle(content_to_key_sample_range)
    content_to_key_sample_range = [content_to_key_sample_range[:IR_task_sample_size_per_epoch]]

    additional_sentences = 4
    include_targets = False
    SP_from_targets = False
    CN_from_targets = False
    num_SP = 2
    num_CN = 2
    num_LP = 2
    hop = 3
    num_nodes_per_hop = 5
    num_workers = 64
    split = "train"
    node_task_save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
    key_to_content_task_save_name = "_".join([SAVE_NAME_BASE, "IR_kc", str(epoch)])
    content_to_key_task_save_name = "_".join([SAVE_NAME_BASE, "IR_ck", str(epoch)])
    generate_default_task(dataset, split, sample_range, node_task_save_name, num_workers, hop, num_nodes_per_hop,
                          node_task_list, additional_sentences, num_SP, num_CN, include_targets,
                          key_to_content_sample_range, key_to_content_task_save_name, content_to_key_sample_range,
                          content_to_key_task_save_name, num_LP, SP_from_targets, CN_from_targets)


if __name__ == "__main__":
    #sample_datasets = ["mag240m", "arxiv", "wiki_graph", "pubmed_node", "ultrachat200k", "wikikg90m"]
    # sample_datasets = ["mag240m", "arxiv", "pubmed_node"]
    # sample_datasets = ["wikikg90m"]
    sample_datasets = ["wikikg90m"]
    SAMPLE_EPOCH = 2
    START_EPOCH = 1
    def random_seed(length):
        random.seed()
        min = 10 ** (length - 1)
        max = 9 * min + (min - 1)
        return random.randint(min, max)
    seed = random_seed(8)
    print("random seed: " + str(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    for dataset in sample_datasets:
        for epoch in range(START_EPOCH, SAMPLE_EPOCH + START_EPOCH):
            globals()["generate_" + dataset](epoch)




