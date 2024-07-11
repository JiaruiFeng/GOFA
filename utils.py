import os
import torch
from torchmetrics import AveragePrecision, AUROC, MeanMetric
import numpy as np
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from torch_scatter import scatter_sum
import re


# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sentence_accuracy(func, output, batch):
    pred = output.logits.reshape(-1, output.logits.size()[-1])
    target = output.answer_id.view(-1)
    return func(pred, target)


def extract_numbers(text):
    """ Extracts all numbers from a given text and returns them as a list of floats. """
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if len(numbers) == 0:
        # default value
        return [0.0]
    return [float(num) for num in numbers]

def sentence_base(func, output, batch):
    pred_text = output.pred_text
    answer = batch.label[batch.label_map.cpu().numpy()].tolist()
    return func(pred_text, answer)


def sentence_mae(func, output, batch):
    pred_text = output.pred_text
    answer = output.answer
    # print(pred_text)
    # print(answer)
    pred_values = [np.mean(extract_numbers(pred)) for pred in pred_text]
    true_values = [np.mean(extract_numbers(true)) for true in answer]
    print("pred", pred_values, "target", true_values)
    return func(torch.tensor(pred_values), torch.tensor(true_values))


def sentence_perplexity(func, output, batch):
    pred = output.logits.unsqueeze(0)
    target = output.answer_id.unsqueeze(0)
    return func(pred, target)


def auc_word(func, output, batch):
    pred = output.logits.reshape(-1, 2, output.logits.size()[-1])[:, 0]
    pos = pred[:, [1939, 3869]]
    pos = torch.softmax(pos, dim=-1)[:, -1]
    target = output.answer_id.view(-1, 2)[:, 0] == 3869
    return func(pos, target.to(torch.long))


def normalized_loss_factory(batch_size, seq_len):
    denom = batch_size * seq_len

    def normalized_loss(func, output, batch):
        sentence_size = len(batch.node_map)
        pred = output.logits.reshape(-1, output.logits.size()[-1])
        target = output.answer_id.view(-1)
        # print(target, pred.sort()[1][:, -1])
        numer = target.ne(-100).sum()
        original_loss = func(pred, target)
        norm_loss = original_loss * numer / (denom * sentence_size)
        return original_loss

    return normalized_loss


def auc_func(func, output, batch):
    pred = output
    label = batch.y.to(pred)
    return func(pred.view(-1), label.view(-1))


def scatter_reg_func(func, output, batch):
    repr_rep = output.repr
    target = output.target[batch.bin_labels[batch.true_nodes_mask].to(torch.bool)]
    return func(repr_rep, target)


class MatCLFunc:
    def __init__(self, temp=0.1, hard_mine_ratio=0.8, sim="mae"):
        self.temp = temp
        if sim == "mae":
            self.loss = torch.nn.L1Loss()
        elif sim == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError("unknown fidelity loss")
        self.hard_mine_ratio = hard_mine_ratio

    def __call__(self, output, batch):
        n_classes = batch.num_classes
        repr_rep = output.repr.repeat_interleave(n_classes, dim=0)
        fidelity_loss = self.loss(output.repr,
                                  output.target[batch.bin_labels[batch.true_nodes_mask].to(torch.bool)].to(torch.float))
        # sim = torch.nn.functional.cosine_similarity(repr_rep.view(repr_rep.size()[0], -1), output.target.view(
        # output.target.size()[0], -1), dim=-1)/self.temp
        sim = torch.abs(repr_rep - output.target).mean(dim=(-1, -2)) / self.temp
        sim = torch.exp(sim)
        class_ind = torch.arange(len(n_classes), device=n_classes.device).repeat_interleave(n_classes, dim=0)
        sim_loss = -torch.log(
            sim[batch.bin_labels[batch.true_nodes_mask].to(torch.bool)] / scatter_sum(sim, class_ind, dim=0)).mean()
        return fidelity_loss


def cl_wrap_func(proc_func, temp=0.1):
    def wrap_func(func, output, batch):
        n_classes = batch.num_classes[0]
        repr_rep = output.repr.repeat_interleave(n_classes, dim=0)
        sim = torch.nn.functional.cosine_similarity(repr_rep.view(repr_rep.size()[0], -1),
                                                    output.target.view(output.target.size()[0], -1), dim=-1) / temp
        sim = sim.view(-1, 1)
        return proc_func(func, sim, batch)

    return wrap_func


class SimAnyAuc(torch.nn.Module):
    def __init__(self, sim_metric="mse"):
        super().__init__()
        if sim_metric == "mse":
            self.loss = torch.nn.MSELoss()
        elif sim_metric == "mae":
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError("unknown fidelity measure")
        self.metric = MeanMetric()

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, output, batch):
        l1loss = self.loss(output.repr, output.target[batch.bin_labels[batch.true_nodes_mask].to(torch.bool)])
        return self.metric(l1loss)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


def mean_func(func, output, batch):
    return func(output, batch)


class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


def scipy_rwpe(data, walk_length):
    row, col = data.edge_index
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value
    adj = to_scipy_sparse_matrix(data.edge_index, edge_attr=value, num_nodes=data.num_nodes)

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1))

    return pe


def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "The molecule is effective to the following assay. " + labels[entry][1][0][:-41])
        label_texts[labels[entry][0] + len(labels)] = (
                "The molecule is not effective to the following assay. " + labels[entry][1][0][:-41])
    return label_texts


def set_mask(data, name, index, dtype=torch.bool):
    mask = torch.zeros(data.num_nodes, dtype=dtype)
    mask[index] = True
    setattr(data, name, mask)
