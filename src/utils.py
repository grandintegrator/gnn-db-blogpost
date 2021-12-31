import dgl
import logging
import torch
from model.dgl.StochasticGCN import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import cat, ones, zeros
from typing import Dict, Any
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch.nn import Sigmoid
import numpy as np


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
    # Compute the AUC per type
    results = {}
    sigmoid = Sigmoid()
    pos_score_edge = sigmoid(pos_score)
    neg_score_edge = sigmoid(neg_score)

    scores = cat([pos_score_edge, neg_score_edge]).detach().numpy()
    labels = cat(
        [ones(pos_score_edge.shape[0]),
         zeros(neg_score_edge.shape[0])]).detach().numpy()

    results['AUC'] = roc_auc_score(labels.astype('int'), scores.squeeze())
    results['AP'] = average_precision_score(labels, scores)
    return results


def plot_tsne_embeddings(graph_embeddings: torch.Tensor,
                         chart_name: str,
                         save_fig: bool) -> None:
    """
    Plots t_sne_embeddings
    """
    # Due to the large size, only plot about ~30% of the embeddings
    indices = np.random.choice(range(graph_embeddings.shape[0]),
                               round(graph_embeddings.shape[0]*0.3),
                               replace=False)
    t_sne_embeddings = (
        TSNE(n_components=2, perplexity=30, method='barnes_hut')
            .fit_transform(graph_embeddings[indices, :])
    )
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(t_sne_embeddings[:, 0],
                t_sne_embeddings[:, 1],
                color='blue',
                linewidths=0.2)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f't-SNE Results for {chart_name}')
    if save_fig:
        plt.savefig(f'data/{chart_name}.png', bbox_inches='tight')
    plt.show()
    plt.close()
