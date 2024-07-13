from TAGLAS import get_task
from typing import (
    Optional,
    Union,
    Callable,
)
from torch.utils.data import Dataset
import torch
import string
import numpy as np

def generate_random_node_order(num_nodes: int):
    return torch.randperm(num_nodes)

def generate_alphabet_id():
    # support graph with less than 26 * 26 nodes.
    alphabet = list(string.ascii_uppercase)
    alphabet_ext = [a + b for a in alphabet for b in alphabet]
    alphabet += alphabet_ext
    return np.array(["[NODEID." + s + "]" for s in alphabet], dtype=object)

def generate_node_id(num_nodes):
    char_ids = generate_alphabet_id()
    node_order = generate_random_node_order(num_nodes)
    return char_ids[node_order.numpy()]


def construct_prompt_graph(num_nodes: int, question_indexs: list, single_direction=False):
    r"""
    Construct and return the prompt graph given num_nodes in the original graph and question_indexs.
    Question_indexs should be a list and each element save the all node indexs of the question.
    """
    num_prompt_nodes = len(question_indexs)
    if num_prompt_nodes == 0:
        return None, None, None
    prompt_node_idx = [num_nodes + i for i in range(num_prompt_nodes)]
    prompt_edge_rows = []
    prompt_edge_cols = []
    prompt_node_index = []
    for i, question_index in enumerate(question_indexs):
        prompt_node_index.append(num_nodes + i)
        prompt_edge_rows.extend(question_index)
        prompt_edge_cols.extend([prompt_node_idx[i] for _ in range(len(question_index))])

    prompt_edges = torch.tensor([prompt_edge_rows, prompt_edge_cols], dtype=torch.long)
    num_prompt_edge = prompt_edges.size(-1)
    prompt_edge_map = torch.zeros(num_prompt_edge, dtype=torch.long)
    prompt_node_index = torch.tensor(prompt_node_index, dtype=torch.long)
    if not single_direction:
        reverse_prompt_edges = prompt_edges[[1, 0], :]
        prompt_edges = torch.cat([prompt_edges, reverse_prompt_edges], dim=-1)
        reverse_edge_map = torch.ones(num_prompt_edge, dtype=torch.long)
        prompt_edge_map = torch.cat([prompt_edge_map, reverse_edge_map], dim=-1)

    return prompt_edges, prompt_edge_map, prompt_node_index


def add_prompt_graph_to_data(
        data,
        prompt_edge_index=None,
        prompt_edge_map=None,
        prompt_edge_text=None,
        prompt_node_text=None):


    if prompt_edge_index is not None:
        assert prompt_edge_map is not None
        assert prompt_edge_map is not None
        assert prompt_edge_text is not None
        data.edge_index = torch.cat([data.edge_index, prompt_edge_index], dim=-1)
        edge_map = data.edge_map
        num_feature_edge_type = edge_map.numel()
        if num_feature_edge_type == 0:
            data.edge_map = torch.cat([edge_map, prompt_edge_map + num_feature_edge_type], dim=-1)
        else:
            data.edge_map = torch.cat([edge_map, prompt_edge_map + edge_map.max()], dim=-1)
        data.edge_attr = np.concatenate([data.edge_attr, prompt_edge_text[prompt_edge_map]], axis=-1)

    if prompt_edge_text is not None:
        data.x = np.concatenate([data.x, prompt_node_text])

    return data


def build_GOFA_task_graph(data, add_prompt_graph=True, is_pretrain=False, single_direction=False, **kwargs):
    r"""GOFA task graph construction function. This function will 1. add node id to nodes in the graph.
    2.specify the Node of generation, either be the target node or add prompt node to the graph.
    if is_pretrain set to False, set to fine-tune model, will assume each graph only contain one QA pair.
    Otherwise, assume the data come with multiple questions and answers. For both two modes, function will automatically
    add prompt node to the graph if question related to more than one node. Otherwise, add prompt node if add_prompt_graph=True.
    """
    num_nodes = data.node_map.size(0)

    node_ids = generate_node_id(num_nodes)
    data.node_ids = node_ids
    # add node id to node text
    for i in range(num_nodes):
        data.x[i] = f"This is node {node_ids[i]}." + data.x[i]

    # Replace placeholder in question and answer with node id
    for i in range(num_nodes):
        for q in range(len(data.question)):
            data.question[q] = data.question[q].replace(f"[NODE_INDEX {i}]", node_ids[i])
        for a in range(len(data.answer)):
            data.answer[a] = data.answer[a].replace(f"[NODE_INDEX {i}]", node_ids[i])
        for l in range(len(data.label)):
            data.label[l] = data.label[l].replace(f"[NODE_INDEX {i}]", node_ids[i])

    # add prompt graph
    prompt_edge_text = np.array(['This edge connects the nodes in graph to a prompt node.',
                                 "This edge connects the prompt node to a node in the graph."], dtype=object)

    if not is_pretrain:
        # fine-tune mode
        question_indexs = [data.target_index.tolist()]

        if len(data.target_index) > 1:
            prompt_edge_index, prompt_edge_map, prompt_node_index = construct_prompt_graph(num_nodes, question_indexs,
                                                                                       single_direction)
            data.question_index = prompt_node_index

        else:
            if add_prompt_graph:
                prompt_edge_index, prompt_edge_map, prompt_node_index = construct_prompt_graph(num_nodes,
                                                                                               question_indexs,
                                                                                               single_direction)
                data.question_index = prompt_node_index

            else:
                prompt_edge_index = prompt_edge_map = None
                data.question_index = data.target_index
        data = add_prompt_graph_to_data(data=data, prompt_edge_index=prompt_edge_index, prompt_edge_map=prompt_edge_map,
                                        prompt_edge_text=prompt_edge_text, prompt_node_text=data.question)
        return data

    else:
        #pretrain mode
        question_indexs = data.target_index
        if add_prompt_graph:
            prompt_edge_index, prompt_edge_map, prompt_node_index = construct_prompt_graph(num_nodes,
                                                                                           question_indexs,
                                                                                           single_direction)
            data.question_index = prompt_node_index
            data = add_prompt_graph_to_data(data=data, prompt_edge_index=prompt_edge_index,
                                            prompt_edge_map=prompt_edge_map, prompt_edge_text=prompt_edge_text,
                                            prompt_node_text=data.question)

        else:
            question_prompt_flag = [True if len(q) > 1 else False for q in question_indexs]
            prompt_question_indexs = [q for q in question_indexs if len(q) > 1]
            prompt_questions = [q for i, q in enumerate(data.question) if question_prompt_flag[i]]

            prompt_edge_index, prompt_edge_map, prompt_node_index = construct_prompt_graph(num_nodes, prompt_question_indexs,
                                                                                           single_direction)

            data = add_prompt_graph_to_data(data=data, prompt_edge_index=prompt_edge_index,
                                            prompt_edge_map=prompt_edge_map, prompt_edge_text=prompt_edge_text,
                                            prompt_node_text=prompt_questions)
            final_question_index = []
            j = 0
            for i in range(len(question_prompt_flag)):
                if question_prompt_flag[i]:
                    final_question_index.append(prompt_node_index[j].item())
                    j += 1
                else:
                    final_question_index.append(question_indexs[i][0])
            data.question_index = torch.tensor(final_question_index, dtype=torch.long)
        return data

















