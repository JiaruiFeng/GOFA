import argparse
import os
from collections import OrderedDict
from datetime import timedelta

import shutil
from lightning.pytorch.loggers import WandbLogger
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from gp.utils.utils import (load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed, )
from gp.lightning.metric import (EvalKit, )
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit, lightning_test
from gp.lightning.module_template import ExpConfig
from lightning_model import GraphTextPredLightning
from gofa_models.model import GOFA
from gofa_models.config import GOFALlamaConfig, GOFAMistralConfig

import torch
from types import SimpleNamespace
import json
import numpy as np
from TAGLAS.data.data import TAGData


def main(params):
    if params.base_llm == 'llama7b':
        from modules.gofa_icae_llama_modeling import ModelArguments, TrainingArguments
        gofa_config = GOFALlamaConfig
    elif params.base_llm == 'mistral7b':
        from modules.gofa_icae_mistral_modeling import ModelArguments, TrainingArguments
        gofa_config = GOFAMistralConfig
    else:
        raise NotImplementedError(params.base_llm + " is not supported. Please choose from: llama7b, mistral7b,")
    if params.mode.endswith("gen"):
        params.last_save = False

    print("available devices: ", torch.cuda.device_count())
    if params.ckpt_save_path is not None:
        date = params.exp_dir.split("/")[-1]
        params.ckpt_save_path = params.ckpt_save_path+"/" + date
    params_dict = vars(params)
    model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), gofa_config(
        num_layers=params.num_layers)
    model_args.dec_lora = params.dec_lora
    model_args.llama_pretrain_checkpoint = params.llama_pretrain_checkpoint
    model_args.mistral_pretrain_checkpoint = params.mistral_pretrain_checkpoint
    training_args.model_max_length = params.llm_max_length
    if params.training_precision == "bf16-mixed":
        training_args.bf16 = True
        gofa_args.llama_dtype = torch.bfloat16
    gofa_args.gnn_mlp_type = params.mlp_type

    model = GOFA(transformer_args=[model_args, training_args, gofa_args], mode=params.mode, base_llm=params.base_llm, save_dir=params.exp_dir)

    if params.load_model:
        print("-"*60+"LOADING"+"-"*60)
        if os.path.isdir(params.load_dir):
            prefix = "_forward_module.model.llm_model.model.icae.base_model.model.model.g_layers."
            state_dict = get_fp32_state_dict_from_zero_checkpoint(params.load_dir)
            partial_dict = OrderedDict()
            for s in state_dict:
                if s.startswith(prefix):
                    partial_dict[s[len(prefix):]] = state_dict[s]
            model.load_partial(state_dict=partial_dict)
        else:
            model.load_partial(load_dir=params.load_dir)

    def textualize_graph(data):
        # mapping from object id to index
        nodes = []
        edges = []
        entities = data['entities']
        relations = data['relations']
        target_id = []
        target_question = []
        target_answer = []
        ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ind = np.arange(len(entities))
        ind = np.stack([(ind/26).astype(int), (ind%26).astype(int)], axis=1)
        char_index = [ascii_uppercase[ind[i][0]] + ascii_uppercase[ind[i][1]] for i in range(len(ind))]
        for nid in entities:
            if "target" in entities[nid] and entities[nid]["target"]["task"] != 'completion':
                node_attr = entities[nid]["desc"]
            else:
                node_attr = f"This is [NODEID.{char_index[int(nid)]}]." + f'{entities[nid]["desc"]}'
            if "target" in entities[nid]:
                target_id.append(int(nid))
                target_answer.append(entities[nid]["target"]["answer"])
                if entities[nid]["target"]["task"] == "completion":
                    target_question.append("")
                else:
                    target_question.append(node_attr)
            nodes.append({'node_id': int(nid), 'node_attr': node_attr})
        for rel in relations:
            src = int(rel["source"])
            dst = int(rel["target"])
            if "general_relation" in rel:
                edge_attr = f'The source node {rel["general_relation"]} target node. Specifically, {rel["specific_relation"]}'
            else:
                edge_attr = rel["specific_relation"]
            edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})
        return nodes, edges, target_id, target_question, target_answer

    def batch_unique_feature(features):
        unique_feature, feature_map = np.unique(features, return_inverse=True)
        feature_map = torch.from_numpy(feature_map).long()

        return unique_feature, feature_map

    model = model.to("cuda")

    while input("Continue generation?") != "no":

        with open("test_graph.json", "r") as f:
            graph = json.load(f)
        t_graph_node, t_graph_edge, target_id, target_question, target_answer = textualize_graph(graph)

        node_text = [None]*len(t_graph_node)
        for i in range(len(t_graph_node)):
            node_text[int(t_graph_node[i]["node_id"])] = t_graph_node[i]["node_attr"]

        edges = []
        edge_texts = []

        for i in range(len(t_graph_edge)):
            edge_texts.append(t_graph_edge[i]["edge_attr"])
            edges.append([t_graph_edge[i]["src"], t_graph_edge[i]["dst"]])

        edges = torch.tensor(edges).T
        node_text = np.array(node_text)
        edge_texts = np.array(edge_texts)
        target_question = np.array(target_question)
        target_answer = np.array(target_answer)
        print(node_text)
        print(edge_texts)
        print(edges)

        unique_node_feature, node_map = batch_unique_feature(node_text)
        unique_edge_feature, edge_map = batch_unique_feature(edge_texts)
        unique_question_feature, q_map = batch_unique_feature(target_question)
        unique_target_feature, a_map = batch_unique_feature(target_answer)

        graph = TAGData(unique_node_feature, node_map, edges, edge_map, unique_edge_feature)
        graph.question = unique_question_feature
        graph.question_map = q_map
        graph.answer = unique_target_feature
        graph.answer_map = a_map
        graph.question_index = torch.tensor(target_id, dtype=torch.long)

        data = graph.to("cuda")
        with torch.no_grad():
            model(data)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line", )

    params = parser.parse_args()
    configs = []
    configs.append(load_yaml(os.path.join(os.path.dirname(__file__), "configs", "default_config.yaml")))

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    # Add for few-shot parameters

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params["root_path"] = mod_params["root_path"] if mod_params["root_path"] else os.environ.get("GGAMA_ROOT_PATH")
    mod_params["data_root_path"] = mod_params["data_root_path"] if mod_params["data_root_path"] else os.environ.get("GGAMA_ROOT_DATA_PATH")
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_float32_matmul_precision("high")
    print(params)
    main(params)
