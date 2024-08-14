from tasks import GOFAPretrainTaskWrapper
import gc

if __name__ == "__main__":
    DATA_ROOT = "/storage1/yinjie.tang/Active/feng.jiarui/TAGDataset"
    SAVE_NAME_BASE = "pretrain"
    SAMPLE_EPOCH = 1
    START_EPOCH = 2
    SAMPLE_DATASETS = ["mag240m"]
    MAG_PRETRAIN_TASK_LIST = ["CS", "CN", "SP"]
    WIKI_PRETRAIN_TASK_LIST = ["DS"]
    ULTRA_PRETRAIN_TASK_LIST = ["DS"]
    WIKIGRAPH_PRETRAIN_TASK_LIST = ["CS"]
    ADDITIONAL_SENTENCES = 3
    WIKIGRAPH_ADDITIONAL_SENTENCES = 1
    SPLIT = "all"
    ULTRA_SPLIT = "train"
    WIKIGRAPH_SPLIT = "train"
    WIKI_SPLIT = "train"
    WIKIGRAPH_LEFT_KEEP_LENGTH = 0
    LEFT_KEEP_LENGTH = 128
    NUM_SP = 3
    NUM_CN = 3
    HOP = 3
    NUM_NODES_PER_HOP = 5
    NUM_WORKERS = 64
    MAG_SAMPLE_SIZE_PER_EPOCH = 300_000
    WIKI_SAMPLE_SIZE_PER_EPOCH = 50_000
    UlTRA_SAMPLE_SIZE_PER_EPOCH = 100_000
    WIKIGRAPH_SAMPLE_SIZE_PER_EPOCH = 50_000
    INCLUDE_TARGETS = True
    WIKIGRAPH_INCLUDE_TARGETS = False

    MAG_SAMPLE_RANGES = [[i * MAG_SAMPLE_SIZE_PER_EPOCH + j for j in range(min(MAG_SAMPLE_SIZE_PER_EPOCH, 5_875_010 - i * MAG_SAMPLE_SIZE_PER_EPOCH))]
                         for i in range(int(5_875_010/MAG_SAMPLE_SIZE_PER_EPOCH) + 1)]
    WIKI_SAMPLE_RANGES = [[i * WIKI_SAMPLE_SIZE_PER_EPOCH + j for j in range(min(WIKI_SAMPLE_SIZE_PER_EPOCH, 5_000_000 - i * WIKI_SAMPLE_SIZE_PER_EPOCH))] for i in range(int(5_000_000/WIKI_SAMPLE_SIZE_PER_EPOCH) + 1)]
    ULTRA_SAMPLE_RANGES = [[i * UlTRA_SAMPLE_SIZE_PER_EPOCH + j for j in range(min(UlTRA_SAMPLE_SIZE_PER_EPOCH, 314950 - i * UlTRA_SAMPLE_SIZE_PER_EPOCH))]
                         for i in range(int(314950/UlTRA_SAMPLE_SIZE_PER_EPOCH) + 1)] ##44993 89986
    WIKIGRAPH_SAMPLE_RANGES = [[i * WIKIGRAPH_SAMPLE_SIZE_PER_EPOCH + j for j in range(min(WIKIGRAPH_SAMPLE_SIZE_PER_EPOCH,
        129_864 - i * WIKIGRAPH_SAMPLE_SIZE_PER_EPOCH))] for i in range(int(129_864/WIKIGRAPH_SAMPLE_SIZE_PER_EPOCH) + 1)]


    for epoch in range(START_EPOCH, START_EPOCH + SAMPLE_EPOCH):
        save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
        num_additional_sentences = ADDITIONAL_SENTENCES
        include_targets = INCLUDE_TARGETS
        left_keep_length = LEFT_KEEP_LENGTH
        split = SPLIT

        for dataset in SAMPLE_DATASETS:
            if dataset == "mag240m":
                sample_range = [MAG_SAMPLE_RANGES[epoch]]
                pretrain_tasks = MAG_PRETRAIN_TASK_LIST
            elif dataset == "wikikg90m":
                sample_range = [WIKI_SAMPLE_RANGES[epoch]]
                pretrain_tasks = WIKI_PRETRAIN_TASK_LIST
                split = WIKI_SPLIT
            elif dataset == "ultrachat200k":
                sample_range = [ULTRA_SAMPLE_RANGES[epoch]]
                pretrain_tasks = ULTRA_PRETRAIN_TASK_LIST
                SPLIT = ULTRA_SPLIT
            elif dataset == "wiki_graph":
                sample_range = [WIKIGRAPH_SAMPLE_RANGES[epoch]]
                pretrain_tasks = WIKIGRAPH_PRETRAIN_TASK_LIST
                include_targets = WIKIGRAPH_INCLUDE_TARGETS
                num_additional_sentences = WIKIGRAPH_ADDITIONAL_SENTENCES
                left_keep_length = WIKIGRAPH_LEFT_KEEP_LENGTH
                split = WIKIGRAPH_SPLIT
            else:
                raise NotImplementedError

            task = GOFAPretrainTaskWrapper(task_names=dataset,
                                           root=DATA_ROOT,
                                           split=split,
                                           sample_size=sample_range,
                                           save_name=save_name,
                                           save_data=True,
                                           from_saved=False,
                                           num_workers=NUM_WORKERS,
                                           hop=HOP,
                                           max_nodes_per_hop=NUM_NODES_PER_HOP,
                                           pretrain_tasks=pretrain_tasks,
                                           num_additional_sentences=num_additional_sentences,
                                           num_SP=NUM_SP,
                                           num_CN=NUM_CN,
                                           include_targets=include_targets,
                                           left_keep_length=left_keep_length,
                                           )
            gc.collect()
