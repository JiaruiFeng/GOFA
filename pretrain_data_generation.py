from tasks import GOFAPretrainTaskWrapper
import gc

if __name__ == "__main__":
    data_root = "./TAGDataset"
    k = 3
    mag_sample_ranges = [
                     [i for i in range(500000)],
                     [i + 500000 for i in range(500000)],
                     [i + 1000000 for i in range(500000)],
                     [i + 1500000 for i in range(500000)],
                     [i + 2000000 for i in range(500000)],
                     [i + 2500000 for i in range(500000)],
                     [i + 3000000 for i in range(500000)],
                     [i + 3500000 for i in range(500000)],
                     [i + 4000000 for i in range(500000)],
                     [i + 4500000 for i in range(500000)],
                     [i + 5000000 for i in range(500000)],
                     [i + 5500000 for i in range(375010)]]

    ultra_sample_ranges = [[i for i in range(40000)],
                           [40000 + i for i in range(40000)],
                           [80000 + i for i in range(40000)],
                           [120000 + i for i in range(40000)],
                           [160000 + i for i in range(40000)],
                           [200000 + i for i in range(40000)],
                           [240000 + i for i in range(40000)],
                           [280000 + i for i in range(40000)],
                           [320000 + i for i in range(40000)],
                           [360000 + i for i in range(40000)]]



    #generate val and test set for ultra QA
    ultra_val = GOFAPretrainTaskWrapper(["ultrachat200k"],
                                                   root=data_root,
                                                   split="all",
                                                   subset_range=[[400000 + i for i in range(20000)]],
                                                   save_name="pretrain_val",
                                                   from_saved=False,
                                                   num_workers=0)

    ultra_test = GOFAPretrainTaskWrapper(["ultrachat200k"],
                                               root=data_root,
                                               split="all",
                                               subset_range=[[420000 + i for i in range(29929)]],
                                               save_name="pretrain_test",
                                               from_saved=False,
                                               num_workers=0)

    del ultra_val, ultra_test
    gc.collect()


    for i in range(3):
        downsample_ranges = [mag_sample_ranges[i], ultra_sample_ranges[i]]
        pretrain_task = GOFAPretrainTaskWrapper(["mag240m", "ultrachat200k"],
                                                root=data_root,
                                                split="all",
                                                subset_ranges=downsample_ranges,
                                                save_name=f"pretrain_subset_{i}",
                                                pretrain_tasks=["CS", "SP", "CN"],
                                                num_workers=64,
                                                num_additional_sentences=3,
                                                num_SP=3,
                                                num_CN=3,
                                                from_saved=False)
        del pretrain_task
        gc.collect()
