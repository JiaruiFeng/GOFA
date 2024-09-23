from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from collections import OrderedDict
import torch

def convert_ckpt(load_dir, save_path):
    prefix = "_forward_module.model.llm_model.model.icae.base_model.model.model.g_layers."
    state_dict = get_fp32_state_dict_from_zero_checkpoint(load_dir)
    partial_dict = OrderedDict()
    for s in state_dict:
        if s.startswith(prefix):
            partial_dict[s[len(prefix):]] = state_dict[s]

    torch.save(partial_dict, save_path)


if __name__ == "__main__":
    load_dir_list = ["saved_exp/2024-09-21 08:31:21.655169_125/GOFA/qp3rxlqu/checkpoints/best_ckptepoch=0-step=1038.ckpt",
                     "saved_exp/2024-09-21 08:31:21.655169_125/GOFA/qp3rxlqu/checkpoints/best_ckptepoch=0-step=2071.ckpt",
                     "saved_exp/2024-09-21 08:31:21.655169_125/GOFA/qp3rxlqu/checkpoints/best_ckptepoch=0-step=3107.ckpt",]

    save_path_list = ["cache_data/mem_ckpt_2_1038.pth",
                      "cache_data/mem_ckpt_2_2071.pth",
                      "cache_data/mem_ckpt_2_3107.pth"]

    for load_dir, save_path in zip(load_dir_list, save_path_list):
        convert_ckpt(load_dir, save_path)