from tasks import GOFAPretrainTaskWrapper
import gc

if __name__ == "__main__":
    DATA_ROOT = "/storage1/yinjie.tang/Active/feng.jiarui/TAGDataset"
    SAMPLE_EPOCH = 1
    START_EPOCH = 0
    SAMPLE_DATASETS = ["wikikg90m"]
    MAG_PRETRAIN_TASK_LIST = ["CS", "CN", "SP"]
    WIKI_PRETRAIN_TASK_LIST = ["CN", "SP", "DS"]
    FB_PRETRAIN_TASK_LIST = ["CN", "SP", "DS"]
    ULTRA_PRETRAIN_TASK_LIST = ["DS"]
    NUM_ADDITIONAL_SENTENCES = 3
    SPLIT = "all"
    NUM_SP = 3
    NUM_CN = 3
    HOP = 3
    NUM_NODES_PER_HOP = 5
    NUM_WORKERS = 96
    MAG_SAMPLE_SIZE_PER_EPOCH = 300_000
    WIKI_SAMPLE_SIZE_PER_EPOCH = 50_000
    UlTRA_SAMPLE_SIZE_PER_EPOCH = 100_000
    FB_SAMPLE_SIZE_PER_EPOCH = 10_000

    SAVE_NAME_BASE = "pretrain"
    MAG_SAMPLE_RANGES = [[i * MAG_SAMPLE_SIZE_PER_EPOCH + j for j in range(MAG_SAMPLE_SIZE_PER_EPOCH)]
                         for i in range(int(5_875_010/MAG_SAMPLE_SIZE_PER_EPOCH))]
    WIKI_SAMPLE_RANGES = [[i * WIKI_SAMPLE_SIZE_PER_EPOCH + j for j in range(WIKI_SAMPLE_SIZE_PER_EPOCH)]
                         for i in range(int(5_000_000/WIKI_SAMPLE_SIZE_PER_EPOCH))]
    ULTRA_SAMPLE_RANGES = [[i * UlTRA_SAMPLE_SIZE_PER_EPOCH + j for j in range(UlTRA_SAMPLE_SIZE_PER_EPOCH)]
                         for i in range(int(449929/UlTRA_SAMPLE_SIZE_PER_EPOCH))]
    FB_SAMPLE_RANGES = [[i * FB_SAMPLE_SIZE_PER_EPOCH + j for j in range(FB_SAMPLE_SIZE_PER_EPOCH)]
                         for i in range(int(200_000/FB_SAMPLE_SIZE_PER_EPOCH))]

    for epoch in range(START_EPOCH, START_EPOCH + SAMPLE_EPOCH):
        save_name = "_".join([SAVE_NAME_BASE, str(epoch)])
        for dataset in SAMPLE_DATASETS:
            if dataset == "mag240m":
                sample_range = [MAG_SAMPLE_RANGES[epoch]]
                pretrain_tasks = MAG_PRETRAIN_TASK_LIST
            elif dataset == "wikikg90m":
                sample_range = [WIKI_SAMPLE_RANGES[epoch]]
                pretrain_tasks = WIKI_PRETRAIN_TASK_LIST
                SPLIT = "train"
            elif dataset == "ultrachat200k":
                sample_range = [ULTRA_SAMPLE_RANGES[epoch]]
                pretrain_tasks = ULTRA_PRETRAIN_TASK_LIST
            elif dataset == "fb15k237":
                sample_range = [FB_SAMPLE_RANGES[epoch]]
                pretrain_tasks = FB_PRETRAIN_TASK_LIST
            else:
                raise NotImplementedError

            task = GOFAPretrainTaskWrapper(task_names=dataset,
                                           root=DATA_ROOT,
                                           split=SPLIT,
                                           sample_size=sample_range,
                                           save_name=save_name,
                                           save_data=True,
                                           from_saved=False,
                                           num_workers=NUM_WORKERS,
                                           hop=HOP,
                                           max_nodes_per_hop=NUM_NODES_PER_HOP,
                                           pretrain_tasks=pretrain_tasks,
                                           num_additional_sentences=NUM_ADDITIONAL_SENTENCES,
                                           num_SP=NUM_SP,
                                           num_CN=NUM_CN,
                                           )
            gc.collect()
