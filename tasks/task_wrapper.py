from gp.lightning.data_template import DatasetWithCollate
from typing import (
    Optional,
    Union,
    Callable,
    Any,
)
from torch import Tensor
from abc import abstractmethod, ABC
from TAGLAS import get_task
from TAGLAS.tasks import GQATask, BaseTask
from TAGLAS.tasks.base import QATask
from TAGLAS.data import TAGData
import numpy as np
import torch
from .build_prompt import build_finetune_task_prompt
from .task_base import build_GOFA_task_graph
from functools import partial
from .pretrain_datasets import get_pretrain_dataset
from .pretrain_tasks import GOFAGraphPretrainTask, GOFALinkPretrainTask, GOFANodePretrainTask

class GOFATaskWrapper(DatasetWithCollate, ABC):
    r"""GOFA task wrapper base class. Use to wrap multiple tasks together.
    Args:
        task_names (Union[list[str], str]): task key to retrieve corresponding tasks.
        root (str): Root directory for saving dataset.
        split (str): Dataset split.
        save_data (bool): If true, will save all generated tasks in root directory.
        from_saved (bool): If ture, will first try to load saved tasks instead of generating a new one.
        save_name (str, optional): If specified, the generated tasks will be saved with the specified name instead of the default name.
        post_funcs (Union[list[Callable], Callable], optional): post-process function for further process each task sample. Will be called in __get_item__.
        filter_func (Callable, optional): data filter function, should return None if data meet filtering condition and original data otherwise.
        sample_size (Union[float, int, list]): sampling parameter for each task. If it is float, will sample the data
            precentagewise. If it is int, sample exact number of data the sample_size is. If it is list, try to sample
            the corresponding  index in the task.
        sample_mode (str): Sample mode for data sampling. Choose from random, balanced, and stratified.
        hop (Union[int, list[int]]): number of hop in subgraph sampling.
        max_nodes_per_hop (Union[int, list[int]]): maximum number of nodes per hop in subgraph sampling.
        num_workers (int): Number of workers when generating the task.
    """
    def __init__(
            self,
            task_names: Union[list[str], str],
            root: str = "TAGDataset",
            split: str = "train",
            save_data: bool = False,
            from_saved: bool = False,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[list[Callable], Callable]] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Union[float, int, list] = 1.0,
            sample_mode: str = "random",
            hop: Union[int, list[int]] = 3,
            max_nodes_per_hop: Union[int, list[int]] = 5,
            num_workers: int = 0,
            **kwargs):
        super().__init__()
        if isinstance(task_names, str):
            task_names = [task_names]
        self.num_tasks = len(task_names)
        self.task_names = task_names
        if isinstance(post_funcs, Callable):
            post_funcs = [post_funcs]
        self.post_funcs = self.__parse_input_args__(post_funcs, self.num_tasks, is_list=True, default_none=True)
        self.roots = self.__parse_input_args__(root, self.num_tasks)
        self.num_workers = self.__parse_input_args__(num_workers, self.num_tasks)
        self.splits = self.__parse_input_args__(split, self.num_tasks)
        self.save_datas = self.__parse_input_args__(save_data, self.num_tasks)
        self.from_saveds = self.__parse_input_args__(from_saved, self.num_tasks)
        self.filter_funcs = self.__parse_input_args__(filter_func, self.num_tasks, default_none=True)
        self.save_names = self.__parse_input_args__(save_name, self.num_tasks, default_none=True)
        self.sample_sizes = self.__parse_input_args__(sample_size, self.num_tasks)
        self.sample_modes = self.__parse_input_args__(sample_mode, self.num_tasks)
        self.hops = self.__parse_input_args__(hop, self.num_tasks)
        self.max_nodes_per_hops = self.__parse_input_args__(max_nodes_per_hop, self.num_tasks)
        self.kwargs = kwargs

        self.task_list = self.__get_task_list__()


        self.task_sizes = np.array([len(t) for t in self.task_list])
        self.ind2task = np.arange(self.num_tasks).repeat(self.task_sizes)
        self.sample_ind = np.concatenate([np.arange(size) for size in self.task_sizes], axis=-1).astype(int)
        self.size_seg = np.cumsum(self.task_sizes)
        self.data_start_index = np.r_[0, self.size_seg[:-1]]


    def __parse_input_args__(self, values: Any, num_task: int, is_list=False, default_none=False) -> list:
        if default_none:
            if values is None:
                return [None for _ in range(num_task)]

        if is_list:
            if isinstance(values[0], list):
                assert len(values) == num_task
                return values
        else:
            if isinstance(values, list):
                assert len(values) == num_task
                return values

        return [values for _ in range(num_task)]

    @abstractmethod
    def __get_task_list__(self, **kwargs):
        pass


    def __getitem__(self, index):
        task_ind = self.ind2task[index]
        task = self.task_list[task_ind]
        data = task[self.sample_ind[index]]
        data.task_idx = task_ind
        return data

    def __len__(self):
        return np.sum(self.task_sizes)

    def collate(self, batch: list[TAGData]):
        float_flag = False
        int_flag = False
        for data in batch:
            if isinstance(data.y, torch.FloatTensor):
                float_flag = True
            elif isinstance(data.y, torch.IntTensor) or isinstance(data.y, torch.LongTensor):
                int_flag = True

        if float_flag and int_flag:
            for data in batch:
                data.y = data.y.float()

        return self.task_list[0].collate(batch)

    def get_collate_fn(self):
        return self.collate


class GOFAPretrainTaskWrapper(GOFATaskWrapper):
    r"""GOFA task wrapper base class. Use to wrap multiple tasks together.
    Args:
        task_names (Union[list[str], str]): task key to retrieve corresponding tasks.
        root (str): Root directory for saving dataset.
        split (str): Dataset split.
        save_data (bool): If true, will save all generated tasks in root directory.
        from_saved (bool): If ture, will first try to load saved tasks instead of generating a new one.
        save_name (str, optional): If specified, the generated tasks will be saved with the specified name instead of the default name.
        post_funcs (Union[list[Callable], Callable], optional): post-process function for further process each task sample. Will be called in __get_item__.
        filter_func (Callable, optional): data filter function, should return None if data meet filtering condition and original data otherwise.
        sample_size (Union[float, int, list]): sampling parameter for each task. If it is float, will sample the data
            precentagewise. If it is int, sample exact number of data the sample_size is. If it is list, try to sample
            the corresponding  index in the task.
        sample_mode (str): Sample mode for data sampling. Choose from random, balanced, and stratified.
        hop (Union[int, list[int]]): number of hop in subgraph sampling.
        max_nodes_per_hop (Union[int, list[int]]): maximum number of nodes per hop in subgraph sampling.
        num_workers (int): Number of workers when generating the task.
        pretrain_tasks (list[str]): The pretrain tasks used for pretraining.
    """

    def __init__(
            self,
            task_names: Union[list[str], str],
            root: Optional[str] = "TAGDataset",
            split: Optional[str] = "all",
            save_data: Optional[bool] = True,
            from_saved: Optional[bool] = True,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[list[Callable], Callable]] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Optional[Union[float, int, list]] = 1.0,
            sample_mode: Optional[str] = "random",
            hop: Optional[Union[int, list[int]]] = 3,
            max_nodes_per_hop: Optional[Union[int, list[int]]] = 5,
            num_workers: Optional[int] = 0,
            pretrain_tasks: list[str] = ["CS"],
            **kwargs):

        if isinstance(task_names, str):
            task_names = [task_names]
        self.num_tasks = len(task_names)
        self.pretrain_tasks = self.__parse_input_args__(pretrain_tasks, self.num_tasks, is_list=True)
        super().__init__(task_names, root, split, save_data, from_saved, save_name, post_funcs, filter_func,
                         sample_size, sample_mode, hop, max_nodes_per_hop, num_workers, **kwargs)


    def __get_pretrain_task__(
            self,
            name: str,
            root: str,
            split: str = "all",
            save_data: bool = True,
            from_saved: bool = True,
            save_name: Optional[str] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Union[int, float] = 1.0,
            sample_mode: str = "random",
            hop: int = 3,
            max_nodes_per_hop: int = 5,
            num_workers: int = 0,
            post_funcs: list[Callable] = None,
            pretrain_tasks: list = ["CS"],
            **kwargs
    ):
        dataset = get_pretrain_dataset(name, root, **kwargs)
        if post_funcs is None:
            post_funcs = []
        add_prompt_graph = False
        if "add_prompt_graph" in kwargs:
            add_prompt_graph = kwargs["add_prompt_graph"]
        single_direction = False
        if "single_direction" in kwargs:
            single_direction = kwargs["single_direction"]

        post_funcs = post_funcs + [partial(build_GOFA_task_graph, is_pretrain=True, add_prompt_graph=add_prompt_graph,
                                           single_direction=single_direction)]

        if name in ["ultrachat200k", "wiki_graph"]:
            task_class = GOFAGraphPretrainTask
        elif name in ["mag240m", "arxiv", "products", "wikics", "cora", "cora_node", "pubmed", "pubmed_node"]:
            task_class = GOFANodePretrainTask
        elif name in ["wikikg90m", "fb15k237", "wn18rr"]:
            task_class = GOFALinkPretrainTask
        else:
            raise NotImplementedError(f"Pretrain task for the dataset {name} is not implemented yet.")

        return task_class(dataset=dataset, split=split, save_data=save_data, from_saved=from_saved,
                        save_name=save_name, post_funcs=post_funcs, filter_func=filter_func,
                        sample_size=sample_size, sample_mode=sample_mode, num_workers=num_workers, hop=hop,
                        max_nodes_per_hop=max_nodes_per_hop, pretrain_tasks=pretrain_tasks, **kwargs)

    def __get_task_list__(self):
        task_list = []
        for i in range(self.num_tasks):
            task_list.append(self.__get_pretrain_task__(
                name=self.task_names[i],
                root=self.roots[i],
                split=self.splits[i],
                save_data=self.save_datas[i],
                from_saved=self.from_saveds[i],
                save_name=self.save_names[i],
                filter_func=self.filter_funcs[i],
                sample_size=self.sample_sizes[i],
                sample_mode=self.sample_modes[i],
                hop=self.hops[i],
                max_nodes_per_hop=self.max_nodes_per_hops[i],
                num_workers=self.num_workers[i],
                pretrain_tasks=self.pretrain_tasks[i],
                **self.kwargs,
                ))
        return task_list


class GOFAFineTuneTaskWrapper(GOFATaskWrapper):
    r"""GOFA task wrapper base class. Use to wrap multiple tasks together.
    Args:
        task_names (Union[list[str], str]): task key to retrieve corresponding tasks.
        root (str): Root directory for saving dataset.
        split (str): Dataset split.
        save_data (bool): If true, will save all generated tasks in root directory.
        from_saved (bool): If ture, will first try to load saved tasks instead of generating a new one.
        save_name (str, optional): If specified, the generated tasks will be saved with the specified name instead of the default name.
        post_funcs (Union[list[Callable], Callable], optional): post-process function for further process each task sample. Will be called in __get_item__.
        filter_func (Callable, optional): data filter function, should return None if data meet filtering condition and original data otherwise.
        sample_size (Union[float, int, list]): sampling parameter for each task. If it is float, will sample the data
            precentagewise. If it is int, sample exact number of data the sample_size is. If it is list, try to sample
            the corresponding  index in the task.
        sample_mode (str): Sample mode for data sampling. Choose from random, balanced, and stratified.
        hop (Union[int, list[int]]): number of hop in subgraph sampling.
        max_nodes_per_hop (Union[int, list[int]]): maximum number of nodes per hop in subgraph sampling.
        num_workers (int): Number of workers when generating the task.
        selection (bool): If true, will generate multiple answer candidates in question and ask the model to select the true one.
        way (int): Number of answer candidates will provide.
        instruct (bool): If true, will also provide a description for each answer candidate.
    """

    def __init__(
            self,
            task_names: Union[list[str], str],
            root: Optional[str] = "TAGDataset",
            split: Optional[str] = "train",
            save_data: Optional[bool] = False,
            from_saved: Optional[bool] = False,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[list[Callable], Callable]] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Optional[Union[float, int, list]] = 1.0,
            sample_mode: Optional[str] = "random",
            hop: Optional[Union[int, list[int]]] = 3,
            max_nodes_per_hop: Optional[Union[int, list[int]]] = 5,
            num_workers: Optional[int] = 0,
            selection: Optional[bool]= True,
            way: Optional[int] = -1,
            instruction: Optional[bool] = True,
            **kwargs):
        if isinstance(task_names, str):
            task_names = [task_names]
        self.num_tasks = len(task_names)

        self.selections = self.__parse_input_args__(selection, self.num_tasks)
        self.ways = self.__parse_input_args__(way, self.num_tasks)
        self.instructions = self.__parse_input_args__(instruction, self.num_tasks)

        super().__init__(task_names, root, split, save_data, from_saved, save_name, post_funcs, filter_func,
                         sample_size, sample_mode, hop, max_nodes_per_hop, num_workers, **kwargs)

    def __get_task_list__(self):
        task_list = []
        for i in range(self.num_tasks):
            prompt_func = partial(build_finetune_task_prompt, task_name=self.task_names[i], way=self.ways[i],
                                  selection=self.selections[i], instruction=self.instructions[i])
            additional_post_funcs = self.post_funcs[i]
            if additional_post_funcs is None:
                additional_post_funcs = []
            post_funcs = additional_post_funcs + [prompt_func, build_GOFA_task_graph]
            task_list.append(get_task(
                name=self.task_names[i],
                task_type="QA",
                root=self.roots[i],
                split=self.splits[i],
                save_data=self.save_datas[i],
                from_saved=self.from_saveds[i],
                save_name=self.save_names[i],
                post_funcs=post_funcs,
                filter_func=self.filter_funcs[i],
                sample_size=self.sample_sizes[i],
                sample_mode=self.sample_modes[i],
                hop=self.hops[i],
                max_nodes_per_hop=self.max_nodes_per_hops[i],
                num_workers=self.num_workers[i],
                **self.kwargs,
                ))

        return task_list






