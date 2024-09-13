from TAGLAS.tasks import NQATask, GQATask, LQATask, BaseTask
from TAGLAS.tasks.process import value_to_tensor, parallel_build_sample_process
from TAGLAS.data import TAGData
from typing import (
    Optional,
    Union,
    Callable
)
from torch import Tensor, LongTensor
import torch
import numpy as np
from .pretrain_task_base import get_pretrain_task


class GOFAGraphPretrainTask(GQATask):
    r"""GOFA graph-level pretrain task class.
    """
    def __init__(
            self,
            pretrain_tasks: list[str] = ["DS"],
            **kwargs):
        self.pretrain_tasks = get_pretrain_task(pretrain_tasks, **kwargs)
        super().__init__(**kwargs)

    def __before_process__(self) -> None:
        super().__before_process__()
        for task in self.pretrain_tasks:
            task.before_process(self)

    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map = self.__process_graph__(index, edge_index, node_map, edge_map)
        target_index = torch.arange(len(node_map))

        question_list = []
        answer_list = []
        label_list = []
        new_target_index_list = []
        for task in self.pretrain_tasks:
            return_dict = task.build_sample(
                task_class=self, node_map=node_map, edge_index=edge_index, target_index=target_index, label_map=label_map, edge_map=edge_map)
            question_list.extend(return_dict["questions"])
            answer_list.extend(return_dict["answers"])
            label_list.extend(return_dict["labels"])
            new_target_index_list.extend(return_dict["target_index"])
            if node_map in return_dict:
                node_map = return_dict["node_map"]
            if "keep_edges" in return_dict:
                keep_edges = return_dict["keep_edges"]
                edge_index = edge_index[:, keep_edges]
                edge_map = edge_map[keep_edges]

        target_index = new_target_index_list
        return TAGData(edge_index=edge_index, node_map=node_map, edge_map=edge_map, target_index=target_index,
                       question=question_list, answer=answer_list, label=label_list)


    def __build_task__(self):
        data_list_ = parallel_build_sample_process(self, graph_level=True)
        data_list = []
        for data in data_list_:
            if "edge_index" in data and getattr(data, "edge_index") is not None and len(data.question) > 0:
                data_list.append(data)
        node_text_list = []
        node_index_list = []
        question_list = []
        question_index_list = []
        answer_list = []
        answer_index_list = []
        label_list = []
        label_index_list = []
        for data in data_list:
            node_text = [self.data.x[i] for i in data.node_map]
            num_nodes = len(node_text)
            node_index_list.append(np.arange(len(node_text_list), num_nodes + len(node_text_list)))
            node_text_list.extend(node_text)

            question = data.question
            num_question = len(question)
            question_index_list.append(np.arange(len(question_list), num_question + len(question_list)))
            question_list.extend(question)

            answer = data.answer
            num_answer = len(answer)
            answer_index_list.append(np.arange(len(answer_list), num_answer + len(answer_list)))
            answer_list.extend(answer)

            label = data.label
            num_label = len(label)
            label_index_list.append(np.arange(len(label_list), num_label + len(label_list)))
            label_list.extend(label)


        unique_question, unique_question_map = np.unique(np.array(question_list, dtype=object), return_inverse=True)
        unique_answer, unique_answer_map = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        unique_label, unique_label_map = np.unique(np.array(label_list, dtype=object), return_inverse=True)
        unique_node_text, unique_node_map = np.unique(np.array(node_text_list, dtype=object), return_inverse=True)

        for i in range(len(data_list)):
            data_list[i].question_map = torch.tensor(unique_question_map[question_index_list[i]]).long()
            data_list[i].answer_map = torch.tensor(unique_answer_map[answer_index_list[i]]).long()
            data_list[i].label_map = torch.tensor(unique_label_map[label_index_list[i]]).long()
            data_list[i].node_map = torch.tensor(unique_node_map[node_index_list[i]]).long()


        self.additional_data = (unique_node_text, unique_label)
        self.question_features = unique_question
        self.answer_features = unique_answer

        return data_list

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[0]

    def __get_label_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[1]

    def __after_process__(self):
        super().__after_process__()
        for task in self.pretrain_tasks:
            task.after_process(self)


class GOFANodePretrainTask(NQATask):
    r"""GOFA node-level pretrain task class. Will load corresponding pretrain tasks given input.
    """
    def __init__(
            self,
            pretrain_tasks: list[str] = ["CS"],
            **kwargs):
        self.pretrain_tasks = get_pretrain_task(pretrain_tasks, **kwargs)
        super().__init__(**kwargs)


    def __process_split_and_label__(self):
        if self.dataset.name == "mag240m" and self.split == "all":
            num_nodes = len(self.dataset.x)
            return (torch.tensor([i for i in range(num_nodes)]), torch.tensor([0 for _ in range(num_nodes)]),
                    torch.tensor([0 for _ in range(num_nodes)]))
        else:
            return super().__process_split_and_label__()

    def __before_process__(self) -> None:
        super().__before_process__()
        for task in self.pretrain_tasks:
            task.before_process(self)

    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map, target_index = self.__process_graph__(index, edge_index, node_map, edge_map)
        target_index = value_to_tensor(target_index)

        question_list = []
        answer_list = []
        label_list = []
        new_target_index_list = []
        for task in self.pretrain_tasks:
            return_dict = task.build_sample(
                task_class=self, node_map=node_map, edge_index=edge_index, target_index=target_index, label_map=label_map, edge_map=edge_map)
            question_list.extend(return_dict["questions"])
            answer_list.extend(return_dict["answers"])
            label_list.extend(return_dict["labels"])
            new_target_index_list.extend(return_dict["target_index"])
            if node_map in return_dict:
                node_map = return_dict["node_map"]
            if "keep_edges" in return_dict:
                keep_edges = return_dict["keep_edges"]
                edge_index = edge_index[:, keep_edges]
                edge_map = edge_map[keep_edges]

        target_index = new_target_index_list
        return TAGData(edge_index=edge_index, node_map=node_map, edge_map=edge_map, target_index=target_index,
                       question=question_list, answer=answer_list, label=label_list)

    def __build_task__(self):
        data_list_ = parallel_build_sample_process(self)
        data_list = []
        for data in data_list_:
            if "edge_index" in data and getattr(data, "edge_index") is not None and len(data.question) > 0:
                data_list.append(data)
        node_text_list = []
        node_index_list = []
        question_list = []
        question_index_list = []
        answer_list = []
        answer_index_list = []
        label_list = []
        label_index_list = []
        for data in data_list:
            node_text = [self.data.x[i] for i in data.node_map]
            num_nodes = len(node_text)
            node_index_list.append(np.arange(len(node_text_list), num_nodes + len(node_text_list)))
            node_text_list.extend(node_text)

            question = data.question
            num_question = len(question)
            question_index_list.append(np.arange(len(question_list), num_question + len(question_list)))
            question_list.extend(question)

            answer = data.answer
            num_answer = len(answer)
            answer_index_list.append(np.arange(len(answer_list), num_answer + len(answer_list)))
            answer_list.extend(answer)

            label = data.label
            num_label = len(label)
            label_index_list.append(np.arange(len(label_list), num_label + len(label_list)))
            label_list.extend(label)


        unique_question, unique_question_map = np.unique(np.array(question_list, dtype=object), return_inverse=True)
        unique_answer, unique_answer_map = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        unique_label, unique_label_map = np.unique(np.array(label_list, dtype=object), return_inverse=True)
        unique_node_text, unique_node_map = np.unique(np.array(node_text_list, dtype=object), return_inverse=True)

        for i in range(len(data_list)):
            data_list[i].question_map = torch.tensor(unique_question_map[question_index_list[i]]).long()
            data_list[i].answer_map = torch.tensor(unique_answer_map[answer_index_list[i]]).long()
            data_list[i].label_map = torch.tensor(unique_label_map[label_index_list[i]]).long()
            data_list[i].node_map = torch.tensor(unique_node_map[node_index_list[i]]).long()


        self.additional_data = (unique_node_text, unique_label)
        self.question_features = unique_question
        self.answer_features = unique_answer

        return data_list

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[0]

    def __get_label_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[1]

    def __after_process__(self):
        super().__after_process__()
        for task in self.pretrain_tasks:
            task.after_process(self)


class GOFALinkPretrainTask(LQATask):
    r"""GOFA link-level pretrain task class.
    """

    def __init__(
            self,
            pretrain_tasks: list[str] = ["CS"],
            **kwargs):
        self.pretrain_tasks = get_pretrain_task(pretrain_tasks, **kwargs)
        super().__init__(**kwargs)

    def __before_process__(self) -> None:
        super().__before_process__()
        for task in self.pretrain_tasks:
            task.before_process(self)

    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map, target_index = self.__process_graph__(index, edge_index, node_map, edge_map)
        target_index = value_to_tensor(target_index)

        question_list = []
        answer_list = []
        label_list = []
        new_target_index_list = []
        for task in self.pretrain_tasks:
            return_dict = task.build_sample(
                task_class=self, node_map=node_map, edge_index=edge_index, target_index=target_index, label_map=label_map, edge_map=edge_map)
            question_list.extend(return_dict["questions"])
            answer_list.extend(return_dict["answers"])
            label_list.extend(return_dict["labels"])
            new_target_index_list.extend(return_dict["target_index"])
            if node_map in return_dict:
                node_map = return_dict["node_map"]
            if "keep_edges" in return_dict:
                keep_edges = return_dict["keep_edges"]
                edge_index = edge_index[:, keep_edges]
                edge_map = edge_map[keep_edges]

        target_index = new_target_index_list
        return TAGData(edge_index=edge_index, node_map=node_map, edge_map=edge_map, target_index=target_index,
                       question=question_list, answer=answer_list, label=label_list)

    def __build_task__(self):
        data_list_ = parallel_build_sample_process(self)
        data_list = []
        for data in data_list_:
            if "edge_index" in data and getattr(data, "edge_index") is not None and len(data.question) > 0:
                data_list.append(data)
        node_text_list = []
        node_index_list = []
        question_list = []
        question_index_list = []
        answer_list = []
        answer_index_list = []
        label_list = []
        label_index_list = []
        for data in data_list:
            node_text = [self.data.x[i] for i in data.node_map]
            num_nodes = len(node_text)
            node_index_list.append(np.arange(len(node_text_list), num_nodes + len(node_text_list)))
            node_text_list.extend(node_text)

            question = data.question
            num_question = len(question)
            question_index_list.append(np.arange(len(question_list), num_question + len(question_list)))
            question_list.extend(question)

            answer = data.answer
            num_answer = len(answer)
            answer_index_list.append(np.arange(len(answer_list), num_answer + len(answer_list)))
            answer_list.extend(answer)

            label = data.label
            num_label = len(label)
            label_index_list.append(np.arange(len(label_list), num_label + len(label_list)))
            label_list.extend(label)


        unique_question, unique_question_map = np.unique(np.array(question_list, dtype=object), return_inverse=True)
        unique_answer, unique_answer_map = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        unique_label, unique_label_map = np.unique(np.array(label_list, dtype=object), return_inverse=True)
        unique_node_text, unique_node_map = np.unique(np.array(node_text_list, dtype=object), return_inverse=True)

        for i in range(len(data_list)):
            data_list[i].question_map = torch.tensor(unique_question_map[question_index_list[i]]).long()
            data_list[i].answer_map = torch.tensor(unique_answer_map[answer_index_list[i]]).long()
            data_list[i].label_map = torch.tensor(unique_label_map[label_index_list[i]]).long()
            data_list[i].node_map = torch.tensor(unique_node_map[node_index_list[i]]).long()


        self.additional_data = (unique_node_text, unique_label)
        self.question_features = unique_question
        self.answer_features = unique_answer

        return data_list

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[0]

    def __get_label_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        return self.additional_data[1]

    def __after_process__(self):
        super().__after_process__()
        for task in self.pretrain_tasks:
            task.after_process(self)
