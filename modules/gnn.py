from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP

from gp.nn.models.util_model import MLP


class GOFAGNNConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, config):
        super().__init__(node_dim=0, aggr="add")

        self.in_dim = config.dim
        self.in_layer = config.mem_token
        self.dropout = config.dropout
        self.head = config.dim

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.lin_qkv = Linear(self.in_dim, self.in_dim * 3)

        self.e_proj = Linear(self.in_dim, self.in_dim * 2, bias=False)
        self.layer_norm_ek = LlamaRMSNorm(self.in_dim)
        self.layer_norm_ev = LlamaRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        if config.gnn_mlp_type == "gp":

            self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=self.dropout,
                          act=config.gnn_hidden_act)
        elif config.gnn_mlp_type == "llama":
            self.ff = LlamaMLP(config)
        else:
            raise NotImplementedError("Unknown mlp type")

        self.x_norm = LlamaRMSNorm(self.in_dim)
        self.xe_norm = LlamaRMSNorm(self.in_dim)

        self.post_gnn_norm = LlamaRMSNorm(self.in_dim)

        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)

        qkv = self.lin_qkv(x)
        query, key, value = torch.chunk(qkv, 3, -1)

        xe = self.e_proj(xe)

        out = self.propagate(edge_index, query=query, key=key, value=value, xe=xe)
        out = self.o_proj(out)

        out = residual + out * self.attn_gate.tanh()

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)

        out = residual + out * self.ff_gate.tanh()

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, xe: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        xe_k, xe_v = torch.chunk(xe, 2, dim=-1)
        key_j = xe_k + key_j
        value_j = value_j + xe_v
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.d_model)

        alpha = softmax(alpha, index, ptr, size_i, dim=0)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j.view(-1, self.d_model)
        out = out * alpha.view(-1, 1)
        out = out.view(-1, self.in_layer, self.in_dim)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
