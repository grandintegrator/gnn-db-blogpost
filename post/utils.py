import dgl
import logging
import torch
# from model.dgl.StochasticGCN import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import cat, ones, zeros
from typing import Dict, Any
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch.nn import Sigmoid
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, aggregator_type):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.n_layers = 2

        self.conv1 = SAGEConv(
            in_feats=in_feats,
            out_feats=hid_feats,
            aggregator_type=aggregator_type
        )
        self.conv2 = SAGEConv(
            in_feats=hid_feats,
            out_feats=out_feats,
            aggregator_type=aggregator_type
        )

    def forward(self, blocks, inputs):
        # inputs are features of nodes
        h = self.conv1(blocks[0], inputs)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h

    def inference(self, g, x, batch_size, device='cpu'):
        """
        Based on the graph and feature set, return the embeddings from
        the final layer of the GraphSAGE model.
        """
        # Compute representations layer by layer
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(g.number_of_nodes(),
                            self.hid_feats
                            if l != self.n_layers - 1
                            else self.out_feats)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                # Copy the features of necessary input nodes to GPU
                h = x[input_nodes].to(device)
                # Compute output.  Note that this computation is the same
                # but only for a single layer.
                h_dst = h[:block.number_of_dst_nodes()]
                h = F.relu(layer(block, (h, h_dst)))
                # Copy to output back to CPU.
                y[output_nodes] = h.cpu()

            x = y

        return y

def create_model(params):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # in_features, hidden_features, out_features, num_classes,
    # etypes):

    model = Model(in_features=params['num_node_features'],
                  hidden_features=params['num_hidden_graph_layers'],
                  out_features=params['num_node_features'],
                  num_classes=params['num_classes'],
                  aggregator_type=params['aggregator_type'])
    return model


def compute_auc_ap(pos_score, neg_score) -> Dict[str, Any]:
  from torch.nn import Sigmoid
  # Compute the AUC per type
  results = {}
  sigmoid = Sigmoid()
  pos_score_edge = sigmoid(pos_score)
  neg_score_edge = sigmoid(neg_score)

  scores = cat([pos_score_edge, neg_score_edge]).detach().numpy()
  labels = cat(
      [ones(pos_score_edge.shape[0]),
       zeros(neg_score_edge.shape[0])]).detach().numpy()

  results['AUC'] = roc_auc_score(labels, scores)
  results['AP'] = average_precision_score(labels, scores)
  return results


def plot_tsne_embeddings(graph_embeddings: torch.Tensor):
    """
    Plots t_sne_embeddings
    """
    import plotly.express as px
    import plotly.io as pio
    import pandas as pd
    pio.templates.default = "plotly_white"

    # Due to the large size, only plot about ~30% of the embeddings
    indices = np.random.choice(range(graph_embeddings.shape[0]),
                               round(graph_embeddings.shape[0]*0.1),
                               replace=False)
    t_sne_embeddings = (
        TSNE(n_components=2, perplexity=30, method='barnes_hut')
            .fit_transform(graph_embeddings[indices, :])
    )

    data = pd.DataFrame({'Dimension 1': t_sne_embeddings[:, 0],
                         'Dimension 2': t_sne_embeddings[:, 1]})

    fig = px.scatter(data, x='Dimension 1', y='Dimension 2')
    fig.update_layout(font_family='Arial',
                      title=f't-SNE',
                      yaxis_title=r"Dimension 1",
                      xaxis_title=r"Dimension 2",
                      font=dict(size=24))
    return fig