import random
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Union,
    overload,
)

import numpy as np
from torch import LongTensor, Tensor
from TAGLAS.tasks import BaseTask
import torch
import networkx as nx
from random import randint
from TAGLAS.tasks.process import value_to_tensor


def get_pretrain_task(tasks: Union[str, list[str]], **kwargs):
    """Get pretrain task function given task name.
    """
    if isinstance(tasks, str):
        tasks = [tasks]

    return_tasks = []
    for task in tasks:
        if task == "CS":
            include_targets = True if "include_targets" not in kwargs else kwargs["include_targets"]
            num_additional_sentences = 0 if "num_additional_sentences" not in kwargs else kwargs["num_additional_sentences"]
            left_keep_length = 0 if "left_keep_length" not in kwargs else kwargs["left_keep_length"]
            return_tasks.append(CompleteSentence(include_targets, num_additional_sentences, left_keep_length))
        elif task == "SP":
            num_SP = 1 if "num_SP" not in kwargs else kwargs["num_SP"]
            SP_from_targets = True if "SP_from_targets" not in kwargs else kwargs["SP_from_targets"]
            return_tasks.append(ShortestPath(num_SP, from_target=SP_from_targets))
        elif task == "CN":
            num_CN = 1 if "num_CN" not in kwargs else kwargs["num_CN"]
            CN_from_targets = True if "CN_from_targets" not in kwargs else kwargs["CN_from_targets"]
            return_tasks.append(CommonNeighbors(num_CN, from_target=CN_from_targets))
        elif task == "DS":
            return_tasks.append(DownstreamTask())
        elif task == "IR":
            content_to_key = False if "content_to_key" not in kwargs else kwargs["content_to_key"]
            num_IR = 1 if "num_IR" not in kwargs else kwargs["num_IR"]
            return_tasks.append(InformationRetrival(num_IR, content_to_key))
        elif task == "LP":
            lp_on_target = False if "lp_on_target" not in kwargs else kwargs["lp_on_target"]
            num_LP = 1 if "num_LP" not in kwargs else kwargs["num_LP"]
            return_tasks.append(LinkPrediction(num_LP, lp_on_target))
        else:
            raise ValueError("Task not defined.")

    return return_tasks


class PretrainTaskBase(ABC):
    r"""Base class for defining a pretrain task.
    The class is designed based on task generation process of TAGLAS. the logic of pretrain tasks generation should be
    defined in three functions, which corresponding to the three functions in the TAGLAS task generation. It will be injected
    into the TAGLAS task generation when be called.
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def before_process(self, task_class, **kwargs):
        pass

    @abstractmethod
    def build_sample(
            self,
            task_class: BaseTask,
            **kwargs):
        pass

    @abstractmethod
    def after_process(self, task_class, **kwargs):
        pass

class DownstreamTask(PretrainTaskBase):
    r"""Use downstream task as pretrain task.
    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def before_process(self, task_class, **kwargs):
        return
    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            label_map: list,
            **kwargs):
        question_map, label_map, answer_map = label_map
        label_map = value_to_tensor(label_map)
        question_map = value_to_tensor(question_map)
        answer_map = value_to_tensor(answer_map)
        question_list = [task_class.question_features[q] for q in question_map]
        answer_list = [task_class.answer_features[a] for a in answer_map]
        label_list = [task_class.data.label[l] for l in label_map]
        target_index = [target_index.tolist()]

        return_dict = {
            "questions": question_list,
            "answers": answer_list,
            "labels": label_list,
            "target_index": target_index,
            "node_map": node_map,
        }
        return return_dict

    def after_process(self, task_class, **kwargs):
        return


class CompleteSentence(PretrainTaskBase):
    r"""Complete sentence pretrain task. If will mask the node text of each target node in the task sample and ask the model
    to complete the mask part. If num_additional_sentences is larger than 0, will additional sample nodes from the input graph.
    Args:
        include_targets (bool): If ture, include all target nodes in the graph for the sentence completion.
        num_additional_sentences (int): The number of additional nodes for the sentence completion besides the target node set.
        left_keep_length (int): The maximum left keep length in complete sentence.
    """
    def __init__(self, include_targets=True, num_additional_sentences: int = 0, left_keep_length: int = 0, **kwargs):
        super().__init__(**kwargs)
        if not include_targets:
            assert num_additional_sentences > 0
        self.include_targets = include_targets
        self.num_additional_sentences = num_additional_sentences
        self.left_keep_length = left_keep_length

    def cut_sentence(self, text_feature):
        words = text_feature.split(" ")
        sentence_length = len(words)
        if sentence_length <= 1:
            return " ".join(words), ""
        elif sentence_length // 2 > self.left_keep_length:
            max_left_length = self.left_keep_length
        else:
            max_left_length = sentence_length // 2
        left_keep_length = randint(0, max_left_length)
        left_words = words[:left_keep_length]
        right_words = words[left_keep_length:]
        return " ".join(left_words), " ".join(right_words)

    def before_process(self, task_class, **kwargs):
        node_texts = task_class.data.x
        left_texts = []
        right_texts = []
        for text in node_texts:
            left_text, right_text = self.cut_sentence(text)
            left_texts.append(left_text)
            right_texts.append(right_text)

        task_class.data.x = task_class.data.x + left_texts
        task_class.right_texts = right_texts
        return
    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            **kwargs):
        prompt_template = "Please complete the sentence of the node [NODE_INDEX <index>]"
        if self.include_targets:
            num_targets = len(target_index)
        else:
            num_targets = 0
            target_index = torch.tensor([], dtype=torch.long)
        num_nodes = node_map.size(0)
        k = max(min(num_nodes - num_targets, self.num_additional_sentences), 0)
        if k > 0:
            target_map = node_map[target_index].tolist()
            select_mask = [False if node in target_map else True for node in node_map]
            selected_index = torch.arange(num_nodes)[select_mask]
            selected_index = selected_index[torch.randperm(len(selected_index))][:k]
            selected_index = selected_index.tolist()
        else:
            selected_index = []

        selected_index = target_index.tolist() + selected_index
        prompt_list = []
        answer_list = []
        total_node_texts = len(task_class.data.x) // 2
        for index in selected_index:
            prompt_list.append(prompt_template.replace("<index>", str(index)))
            answer_list.append(task_class.right_texts[node_map[index]])
            node_map[index] = total_node_texts + node_map[index]

        # task_class.data.x = node_features + left_texts
        target_index = [[index] for index in selected_index]
        return_dict = {
            "questions": prompt_list,
            "answers": answer_list,
            "labels": answer_list,
            "target_index": target_index,
            "node_map": node_map,
        }
        return return_dict

    def after_process(self, task_class, **kwargs):
        task_class.right_texts = None


class ShortestPath(PretrainTaskBase):
    r"""Shortest path pretrain task. It will generate task to ask model the shortest path and SPD between node pairs in the task graph sample.
    Args:
        num_SP (int): Number of node pair to include.
        from_target (bool): If ture, one node in the node pair will be the target node.
    """
    def __init__(self, num_SP: int = 1, from_target=True, **kwargs):
        super().__init__(**kwargs)
        self.num_SP = num_SP
        self.from_target = from_target

    def compute_shortest_paths(self, i, j, edge_index):
        # Compute the shortest path between node i and node j
        # return all shortest paths and distance.
        G = nx.from_edgelist(edge_index.numpy().T)
        # check whether i or j is an isolated node or no path exists between i and j
        has_path = i in G and j in G and nx.has_path(G, i, j)
        if has_path:
            path_list = nx.all_shortest_paths(G, i, j)
            path_list = torch.tensor(list(path_list))
            return path_list, path_list.size(-1) - 1
        else:
            return [], 'inf'

    def path_list_to_text(self, path_list, special_token="[NODE_INDEX <index>]"):
        if isinstance(path_list, Tensor):
            path_list = path_list.tolist()
        path_str_list = []
        for path in path_list:
            path = [special_token.replace("<index>", str(p)) for p in path]
            path_str_list.append(" -> ".join(path))
        return "; ".join(path_str_list)

    def before_process(self, task_class, **kwargs):
        return

    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            **kwargs):

        prompt_template = ("Compute the shortest path distance between the node [NODE_INDEX <i>] and node "
                           "[NODE_INDEX <j>] and generate all shortest paths from [NODE_INDEX <i>] to [NODE_INDEX <j>]. "
                           "Please separate nodes in path with ->. If multiple paths exist, generate all of them with "
                           "an ascending order of node index and separate different paths with ;.")

        num_nodes = len(node_map)
        question_list = []
        answer_list = []
        label_list = []
        target_index_list = []
        non_target_index = [i for i in range(num_nodes) if i not in target_index]
        for _ in range(self.num_SP):
            if self.from_target:
                i = target_index[torch.randperm(len(target_index))[0]].item()
                if len(non_target_index) == 0:
                    non_target_index = [i for i in range(num_nodes) if i not in target_index]
                j = random.choice(non_target_index)
                non_target_index.remove(j)
            else:
                node_pair = torch.randperm(num_nodes)[:2]
                i, j = node_pair[0].item(), node_pair[1].item()
            target_index_list.append([i, j])
            question = prompt_template.replace("<i>", str(i)).replace("<j>", str(j))
            path_list, spd = self.compute_shortest_paths(i, j, edge_index)
            path_text =self.path_list_to_text(path_list)
            if len(path_list):
                answer = f"The shortest path distance is {str(spd)}. Shortest paths: " + path_text + "."
            else:
                answer = f"The shortest path distance is {str(spd)}. There is no path between [NODE_INDEX {i}] and [NODE_INDEX {j}]."
            label = str(spd)
            question_list.append(question)
            answer_list.append(answer)
            label_list.append(label)

        return_dict = {
            "questions": question_list,
            "answers": answer_list,
            "labels": label_list,
            "target_index": target_index_list,
        }
        return return_dict

    def reorder_paths(self, path_text):
        path_texts = path_text.split(": ")
        path_text_list = path_texts[-1][:-1].split("; ")
        path_text_list.sort()
        new_answer = path_texts[0] + ": " + "; ".join(path_text_list) + "."
        return new_answer
    def after_process(self, task_class, **kwargs):
        new_answer_list = []
        for answer in task_class.answer_features:
            if answer.startswith("The shortest path distance is"):
                new_answer_list.append(self.reorder_paths(answer))
            else:
                new_answer_list.append(answer)
        task_class.answer_features = np.array(new_answer_list, dtype=object)

        return


class CommonNeighbors(PretrainTaskBase):
    r"""Common neighbor pretrain task. It will generate task to ask model the number and exact list of common neighbors
        between node pairs in the task graph sample.
    Args:
        num_CN (int): Number of node pair to include.
        from_target (bool): If ture, one node in the node pair will be the target node.
    """
    def __init__(self, num_CN: int=1, from_target=True, **kwargs):
        self.num_CN = num_CN
        self.from_target = from_target

    def compute_common_neighbors(self, i, j, edge_index):
        i_neighbors = edge_index[1, edge_index[0] == i]
        j_neighbors = edge_index[1, edge_index[0] == j]
        cns = i_neighbors[torch.isin(i_neighbors, j_neighbors)]
        return cns

    def nodes_to_text(self, node_order_list, special_token="[NODE_INDEX <index>]"):
        node_order_list = [special_token.replace("<index>", str(node)) for node in node_order_list]
        return "; ".join(node_order_list)

    def before_process(self, task_class, **kwargs):
        return


    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            **kwargs):

        prompt_template = ("Is there any common neighbors between the node [NODE_INDEX <i>] and node [NODE_INDEX <j>]? "
                           "If exist, please give the total number and list all common neighbors with ascending order "
                           "of node, separate nodes with ;.")

        num_nodes = len(node_map)
        question_list = []
        answer_list = []
        label_list = []
        target_index_list = []
        non_target_index = [i for i in range(num_nodes) if i not in target_index]
        for _ in range(self.num_CN):
            if self.from_target:
                i = target_index[torch.randperm(len(target_index))[0]].item()
                if len(non_target_index) == 0:
                    non_target_index = [i for i in range(num_nodes) if i not in target_index]
                j = random.choice(non_target_index)
                non_target_index.remove(j)
            else:
                node_pair = torch.randperm(num_nodes)[:2]
                i, j = node_pair[0].item(), node_pair[1].item()
            target_index_list.append([i, j])
            question = prompt_template.replace("<i>", str(i)).replace("<j>", str(j))
            cns = self.compute_common_neighbors(i, j, edge_index)
            cns = cns.tolist()
            if len(cns) == 0:
                answer = "There is no common neighbors between two nodes."
            else:
                cns_text = self.nodes_to_text(cns)
                answer = f"There are {str(len(cns))} common neighbors between two nodes, including {cns_text}."
            label = str(len(cns))
            question_list.append(question)
            answer_list.append(answer)
            label_list.append(label)

        return_dict = {
            "questions": question_list,
            "answers": answer_list,
            "labels": label_list,
            "target_index": target_index_list,
        }
        return return_dict

    def reorder_nodes(self, text):
        node_texts = text.split("including ")
        node_text_list = node_texts[-1][:-1].split("; ")
        node_text_list.sort()
        new_answer = node_texts[0] + "including " + "; ".join(node_text_list) + "."
        return new_answer

    def after_process(self, task_class, **kwargs):
        new_answer_list = []
        for answer in task_class.answer_features:
            if "common neighbors between two nodes, including " in answer:
                new_answer_list.append(self.reorder_nodes(answer))
            else:
                new_answer_list.append(answer)
        task_class.answer_features = np.array(new_answer_list, dtype=object)
        return


def single_node_graph_complete_sentence(data, **kwargs):
    """Only work if the input data are complete sentence task with no additional sentence
    """
    target_index = data.target_index[0]
    data.x = data.x[target_index]
    data.node_map = torch.tensor([0], dtype=torch.long)
    data.edge_index = torch.tensor([[], []], dtype=torch.long)
    data.edge_map = torch.tensor([], dtype=torch.long)
    data.edge_attr = np.array([], dtype=object)
    #data.question = np.array(["Please complete the sentence of the target node."], dtype=object)
    data.question[0] = data.question[0].replace(f"[NODE_INDEX {target_index[0]}]", f"[NODE_INDEX 0]")
    data.question = data.question[[0]]
    data.question_map = torch.tensor([0], dtype=torch.long)
    data.answer = data.answer[[0]]
    data.answer_map = torch.tensor([0], dtype=torch.long)
    data.label = data.label[[0]]
    data.label_map = torch.tensor([0], dtype=torch.long)
    data.target_index = [[0]]
    return data


class InformationRetrival(PretrainTaskBase):
    r"""Information retrival task. The task contains two modes. In default mode, the task ask model to retrival content based on node ID.
    If set content_to_key to True, the task as model to retrival node ID based on content.
    Args:
        num_IR (int): Number of information retrival task to generate.
        content_to_key (bool): If true, the task become ask model the node ID given node content.
    """
    def __init__(self, num_IR, content_to_key=False, **kwargs):
        self.num_IR = num_IR
        self.content_to_key = content_to_key

    def before_process(self, task_class, **kwargs):
        return


    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            **kwargs):

        content_to_key_prompt = "Please give the ID of the node which contains the following content: "
        key_to_content_prompt = "Please output the content of node [NODE_INDEX <i>]."


        num_nodes = len(node_map)
        question_list = []
        answer_list = []
        label_list = []
        target_index_list = []
        for _ in range(self.num_IR):
            # random select a numer k in (2, num_nodes)
            k = randint(2, num_nodes)
            # random select k nodes from graph.
            selected_index = torch.randperm(num_nodes)[:k]
            target_index_list.append(selected_index.tolist())
            # random select node to be the answer node.
            answer_index = selected_index[torch.randperm(k)[0]]
            if self.content_to_key:
                question = content_to_key_prompt + task_class.data.x[node_map[answer_index]]
                answer = "[NODE_INDEX <i>]".replace("<i>", str(answer_index.item()))
            else:
                question = key_to_content_prompt.replace("<i>", str(answer_index.item()))
                answer = task_class.data.x[node_map[answer_index]]
            label = answer
            question_list.append(question)
            answer_list.append(answer)
            label_list.append(label)

        return_dict = {
            "questions": question_list,
            "answers": answer_list,
            "labels": label_list,
            "target_index": target_index_list,
        }
        return return_dict

    def after_process(self, task_class, **kwargs):
        return


class LinkPrediction(PretrainTaskBase):
    r"""Link prediction task. Randomly select one edge, remove the edge and ask the model to predict the content in the edge.

    Args:
        num_LP (int): Number of link prediction task to generate.
        lp_on_target (bool): If true, construct LP task on target node. Should be set to true only when base task are link task.
    """
    def __init__(self, num_LP, lp_on_target=False, **kwargs):
        self.num_LP = num_LP
        self.lp_on_target = lp_on_target

    def before_process(self, task_class, **kwargs):
        return

    def build_sample(
            self,
            task_class: BaseTask,
            node_map: LongTensor,
            edge_index: LongTensor,
            target_index: LongTensor,
            **kwargs):
        edge_map = kwargs["edge_map"]
        prompt = ("There exist one edge between source node [NODE_INDEX <i>] and target node [NODE_INDEX <j>]. "
                  "Could you generate correct content in the edge based on information in two nodes?")
        num_edges = edge_index.size(-1)
        question_list = []
        answer_list = []
        label_list = []
        target_index_list = []
        # random select num_LP edges
        selected_indexs = torch.randperm(num_edges)[:self.num_LP]
        remove_edges = []
        for index in selected_indexs:
            edge = edge_index[:, index]
            remove_edges.extend([edge.tolist(), edge[[1, 0]].tolist()])
            target_index_list.append(edge.tolist())
            question_list.append(prompt.replace("<i>", str(edge[0].item())).replace("<j>", str(edge[1].item())))
            answer_list.append(task_class.data.edge_attr[edge_map[index]])

        keep_edges = torch.tensor([i for i in range(num_edges) if edge_index[:, i].tolist()
                                   not in remove_edges], dtype=torch.long)
        return_dict = {
            "questions": question_list,
            "answers": answer_list,
            "labels": label_list,
            "target_index": target_index_list,
            "keep_edges": keep_edges
        }
        return return_dict

    def after_process(self, task_class, **kwargs):
        return
