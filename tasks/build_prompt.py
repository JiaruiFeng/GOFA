from random import shuffle, randint
from typing import (
    Union,
)

import numpy as np



def sample_k_labels_with_true(
        full_label_list: Union[list[str], np.ndarray],
        true_label_idx: int,
        way: int = 10) -> list[str]:
    r"""
    Sample k labels from a complete label list containing the true label, ensuring that the true label is included
    in the sampled set.

    Parameters:
    full_label_list (list[str]): A complete list of multiple labels.
    true_label (int): The index of the true label.
    way (int, optional): The number of labels to be sampled, defaulting to 10.

    Returns:
    list[str]: A list of sampled labels, guaranteed to contain the true label.
    """

    # Ensure the requested number of labels to sample does not exceed the length of the complete label list
    if way == -1 or len(full_label_list) < way:
        return full_label_list
        # way = len(full_label_list)
    if isinstance(full_label_list, np.ndarray):
        full_label_list = full_label_list.tolist()
    true_label = full_label_list[true_label_idx]
    label_sample_list = full_label_list[:true_label_idx] + full_label_list[true_label_idx + 1:]
    shuffle(label_sample_list)
    label_sample_list = label_sample_list[:way - 1]
    label_sample_list.insert(randint(0, way), true_label)
    shuffle(label_sample_list)
    return label_sample_list


def build_finetune_task_prompt(data, task_class, task_name, way=5, selection=True, instruction=True, **kwargs):

    if not selection and not instruction:
        return default_prompt(data, task_class)

    else:
        if f"{task_name}_prompt" in globals():
            return globals()[f"{task_name}_prompt"](
                data, task_class=task_class, selection=selection, way=way, instruction=instruction, **kwargs)
        else:
            return default_prompt(data, task_class)


def default_prompt(data, task_class, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    # directly use the initial prompt, mainly for supervised fine-tuning
    data.question = graph_description + question
    return data

def cora_node_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label[:-2]
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You are an expert in computer science. You need to choose the correct paper category based "
                            "on the paper content and its co-citation network. For example, ") + "; ".join(
            (f"if the paper [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def cora_link_prompt(data, task_class, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    if instruction:
        instruction_prompt = (f"You are a computer science expert tasked with determining whether two given papers "
                              f"[NODE_INDEX {data.target_index[0].item()}] and [NODE_INDEX {data.target_index[1].item()}] "
                              f"in the computer science domain are co-cited by another paper based on their content and"
                              f" network characteristics. If two papers are from the same category, the content of the "
                              f"two papers is similar, the shortest path distance between the two papers is small, or "
                              f"the papers have a large number of common neighbors in the citation network, choose Yes. "
                              f"If two papers are different, the shortest path distance between the two papers is large,"
                              f" or the npapers do not have many common neighbors in the citation network, choose No. ")
    else:
        instruction_prompt = ""
    data.question = graph_description + instruction_prompt + question
    return data


def pubmed_node_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label[:-2]
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You are an expert in diabetes mellitus. You need to choose the correct paper category based "
                            "on the paper content and its co-citation network. For example, ") + "; ".join(
            (f"if the paper [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data



def pubmed_link_prompt(data, task_class, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    if instruction:
        instruction_prompt = (f"You are a diabetes mellitus expert tasked with determining whether two given papers "
                              f"[NODE_INDEX {data.target_index[0].item()}] and [NODE_INDEX {data.target_index[1].item()}] "
                              f"in the diabetes mellitus domain are co-cited by another paper based on their content and"
                              f" network characteristics. If two papers are from the same category, the content of the "
                              f"two papers is similar, the shortest path distance between the two papers is small, or "
                              f"the papers have a large number of common neighbors in the citation network, choose Yes. "
                              f"If two papers are different, the shortest path distance between the two papers is large,"
                              f" or the npapers do not have many common neighbors in the citation network, choose No. ")
    else:
        instruction_prompt = ""
    data.question = graph_description + instruction_prompt + question
    return data

def arxiv_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label[:-2]
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You are an expert in computer science. You need to choose the correct paper category based "
                            "on the paper content and its citation network. For example, ") + "; ".join(
            (f"if the paper [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def wikics_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You are an expert in computer science. You need to choose the correct category "
                              "of Wikipedia term based on the term content. For example, ") + "; ".join(
            (f"if the term [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def products_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You need to choose the correct category of the target product based on the term content. For example, ") + "; ".join(
            (f"if the product [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def fb15k237_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = (f"You are an expert in knowledge graph reasoning. You need to choose the correct relation type between two "
                    f"target entities [NODE_INDEX {data.target_index[0].item()}] and [NODE_INDEX {data.target_index[1].item()}] "
                    f"based on their existing relations. For example, ") + "; ".join(
            (f"if " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def wn18rr_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = (f"You are an expert in the English language. You need to choose the correct relationship between two English words "
                    f"target words [NODE_INDEX {data.target_index[0].item()}] and [NODE_INDEX {data.target_index[1].item()}] "
                    f"based on the meaning of the words. For example, ") + "; ".join(
            (f"if " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data

def ml1m_prompt(data, task_class, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    if instruction:
        instruction_prompt = (f"You are a movie recommendation expert working on predicting how much the user "
                    f"[NODE_INDEX {data.target_index[0].item()}] will like the movie [NODE_INDEX {data.target_index[1].item()}]. "
                    f"Score ranges from 1 to 5, a higher score indicates greater preference. For example, if many users "
                    f"have given high scores to a movie or this user typically gives high scores to similar types of "
                    f"movies, choose a higher score; if many users have given low scores to the movie or this user "
                    f"generally rates similar types of movies lower, choose a lower score. ")
    else:
        instruction_prompt = ""
    data.question = graph_description + instruction_prompt + question
    return data

def expla_graph_prompt(data, task_class, instruction=True, **kwargs):
    question = data.question
    question[0] = question[0].replace("\n", " ")
    graph_description = task_class.dataset.graph_description
    if instruction:
        instruction_prompt = ("You are a logic expert tasked with analyzing the logical relationship between two "
                              "arguments related to connected entities. Determine if the arguments support or counter "
                              "each other based on their logical coherence. If there is no logical conflict between "
                              "the two arguments and they are in agreement, choose Support; if the arguments exhibit "
                              "a logical conflict or contradiction, choose Counter. ")
    else:
        instruction_prompt = ""
    data.question = graph_description + instruction_prompt + question
    return data

def scene_graph_prompt(data, task_class, **kwargs):
    return default_prompt(data, task_class, **kwargs)


def mag240m_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = ("You are an expert in academic research. You need to choose the correct paper category based "
                            "on the paper content and its citation network. For example, ") + "; ".join(
            (f"if the paper [NODE_INDEX {data.target_index.item()}] " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data


def wikikg90m_prompt(data, task_class, selection=True, way=-1, instruction=True, **kwargs):
    question = data.question
    graph_description = task_class.dataset.graph_description
    label_list = task_class.dataset.label
    if selection:
        label_selection_list = sample_k_labels_with_true(label_list, data.label_map.item(), way=way)
        selection_prompt = " Choose from the following: " + "; ".join(label_selection_list) + "."
    else:
        label_selection_list = label_list
        selection_prompt = ""

    if instruction:
        label_desc = task_class.dataset.side_data.label_description
        instruction_prompt = (f"You are an expert in knowledge graph reasoning. You need to choose the correct relation type between two "
                    f"target entities [NODE_INDEX {data.target_index[0].item()}] and [NODE_INDEX {data.target_index[1].item()}] "
                    f"based on their existing relations. For example, ") + "; ".join(
            (f"if two entities involve " + label_desc[label][:-1].strip(".").lower().replace(".", ",")
             + ", choose " + label) for label in label_selection_list) + ". "
    else:
        instruction_prompt = ""

    data.question = graph_description + instruction_prompt + question + selection_prompt
    return data
