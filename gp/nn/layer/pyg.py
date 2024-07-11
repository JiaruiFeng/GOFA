import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops

from typing import Optional, Tuple, Union

from gp.nn.models.util_model import MLP

from transformers import LlamaConfig, MistralConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, torch.Tensor):
        return edge_index[:, edge_mask]


class RGCNEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_relations: int, aggr: str = "mean", **kwargs, ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(torch.empty(self.num_relations, in_channels, out_channels))

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: OptTensor, xe: OptTensor, edge_index: Adj, edge_type: OptTensor = None, ):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x, xe=xe[edge_mask])
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return (x_j + xe).relu()


class RGATEdgeConv(RGCNEdgeConv):
    def __init__(self, in_channels: int, out_channels: int, num_relations: int, aggr: str = "sum", heads=8,
                 add_self_loops=False, share_att=False, **kwargs, ):
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.share_att = share_att
        super().__init__(in_channels, out_channels, num_relations, aggr, node_dim=0, **kwargs, )
        self.lin_edge = nn.Linear(self.in_channels, self.out_channels)
        assert self.in_channels % heads == 0
        self.d_model = self.in_channels // heads
        if self.share_att:
            self.att = Parameter(torch.empty(1, heads, self.d_model))
        else:
            self.att = Parameter(torch.empty(self.num_relations, heads, self.d_model))

        glorot(self.att)

    def forward(self, x: OptTensor, xe: OptTensor, edge_index: Adj, edge_type: OptTensor = None, ):
        out = torch.zeros((x.size(0), self.out_channels), device=x.device)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, xe, fill_value="mean", num_nodes=num_nodes)

        x_ = x.view(-1, self.heads, self.d_model)
        xe_ = self.lin_edge(xe).view(-1, self.heads, self.d_model)

        for i in range(self.num_relations):
            edge_mask = edge_type == i
            if self.add_self_loops:
                edge_mask = torch.cat([edge_mask, torch.ones(num_nodes, device=edge_mask.device).bool(), ])

            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x_, xe=xe_[edge_mask], rel_index=i)
            h = h.view(-1, self.in_channels)
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe, rel_index, index, ptr, size_i):
        # x_j has shape [E, out_channels]
        x = F.leaky_relu(x_j + xe)
        if self.share_att:
            att = self.att

        else:
            att = self.att[rel_index: rel_index + 1]

        alpha = (x * att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        return (x_j + xe) * alpha.unsqueeze(-1)


class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_dim: int, in_layer: int, head: int, add_self_loops=True, dropout: float = 0., **kwargs, ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.in_layer = in_layer
        self.dropout = dropout
        self.head = head
        self.layer_idx = kwargs["layer_idx"]

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.lin_qkv = Parameter(torch.empty(self.in_dim, self.in_dim * 3))
        self.bias_qkv = Parameter(torch.empty(1, self.in_dim * 3))

        self.e_proj = Linear(self.in_dim, self.in_dim * 2, bias=False)
        self.layer_norm_ek = LlamaRMSNorm(self.in_dim)
        self.layer_norm_ev = LlamaRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        llama_config = LlamaConfig(hidden_size=self.in_dim, intermediate_size=self.in_dim * 2,
                                   num_attention_heads=self.head)
        self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=dropout)
        self.ff2 = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=dropout)

        self.x_norm = LlamaRMSNorm(self.in_dim)
        self.xe_norm = LlamaRMSNorm(self.in_dim)

        self.post_gnn_norm = LlamaRMSNorm(self.in_dim)
        self.pre_att_norm = LlamaRMSNorm(self.in_dim)
        self.post_att_norm = LlamaRMSNorm(self.in_dim)

        self.self_att = LlamaAttention(llama_config, self.layer_idx)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_uniform_(self.lin_qkv)
        nn.init.constant_(self.bias_qkv, 0)
        self.o_proj.reset_parameters()  # self.ff.reset_parameters()  # self.x_norm.reset_parameters()  #  #
        # self.xe_norm.reset_parameters()  # self.layer_norm1.reset_parameters()  # self.layer_norm2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        # x = x.permute(1, 0, 2)
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)
        # x_ = x.permute(1, 0, 2)
        # qkv = torch.bmm(x_, self.lin_qkv) + self.bias_qkv
        # qkv = qkv.permute(1, 0, 2)
        # qkv = qkv.view(-1, self.in_layer, self.in_dim * 3)
        qkv = x @ self.lin_qkv + self.bias_qkv
        query, key, value = torch.chunk(qkv, 3, -1)

        xe = self.e_proj(xe)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value, xe=xe)
        out = self.o_proj(out)

        out = residual + out

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)

        out = residual + out

        residual = out

        out = self.pre_att_norm(out)

        out = self.self_att(out)[0]

        out = residual + out

        residual = out

        out = self.post_att_norm(out)
        out = self.ff2(out)
        out = residual + out

        return out.view(-1, self.in_layer * self.in_dim)

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


