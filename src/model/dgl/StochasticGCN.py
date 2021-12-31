import dgl
import torch.nn as nn
from model.dgl.layers import GraphSAGE  # StochasticTwoLayerGCN
import torch.nn.functional as F
from torch import cat


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(in_features * 2, 1)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes,
                 aggregator_type):
        super().__init__()
        self.gcn = GraphSAGE(in_features, hidden_features, out_features,
                             aggregator_type)
        self.pred = MLPPredictor(out_features)

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score

    def get_embeddings(self, g, x, batch_size, device):
        return self.gcn.inference(g, x, batch_size, device)
