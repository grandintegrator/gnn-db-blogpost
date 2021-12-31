import dgl
import tqdm
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch import cat, ones, zeros
from typing import Dict, Any
from model.dgl.StochasticGCN import ScorePredictor
from utils import plot_tsne_embeddings, compute_auc_ap
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


class Trainer(object):
    def __init__(self,
                 params: Dict[str, Any],
                 model: torch.nn.Module,
                 train_data_loader: dgl.dataloading.EdgeDataLoader,
                 validation_data_loader: dgl.dataloading.EdgeDataLoader,
                 training_graph: dgl.DGLGraph):
        self.params = params
        self.model = model
        self.training_graph = training_graph
        self.train_data_loader = train_data_loader
        self.predictor = ScorePredictor().to(self.params['device'])
        self.opt = None
        self.sigmoid = Sigmoid()

    def __repr__(self):
        return "Training Manager class"

    def make_optimiser(self):
        # Fixing parameter types because Box doesn't do this naturally.
        self.params['lr'] = float(self.params['lr'])
        self.params['l2_regularisation'] = \
            float(self.params['l2_regularisation'])

        # Could probably turn this into a function if we want to try others
        if self.params['optimiser'] == 'SGD':
            self.opt = optim.SGD(self.model.parameters(), lr=self.params['lr'],
                                 momentum=self.params['momentum'],
                                 weight_decay=self.params['l2_regularisation'])
        elif self.params['optimiser'] == 'Adam':
            self.opt = optim.Adam(self.model.parameters(),
                                  lr=self.params['lr'],
                                  weight_decay=self.params['l2_regularisation'])

    def compute_loss(self, pos_score, neg_score):
        # For computing the pos and negative score just for the inference edge
        pos_score_edge = self.sigmoid(pos_score)
        neg_score_edge = self.sigmoid(neg_score)
        n = pos_score_edge.shape[0]

        if self.params['loss'] == 'margin':
            margin_loss = (
                (neg_score_edge.view(n, -1) -
                 pos_score_edge.view(n, -1) + 1)
                    .clamp(min=0).mean()
            )
            return margin_loss
        elif self.params['loss'] == 'binary_cross_entropy':
            scores = cat([pos_score_edge, neg_score_edge])
            labels = \
                cat([ones(pos_score_edge.shape[0]),
                     zeros(neg_score_edge.shape[0])])
            scores = scores.view(len(scores), -1).mean(dim=1)  # Fixing dims
            binary_cross_entropy = \
                F.binary_cross_entropy_with_logits(scores, labels)
            return binary_cross_entropy

    def train_epochs(self):
        with tqdm.tqdm(self.train_data_loader) as tq:
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(tq):
                # For transferring to CUDA
                if self.params['device'] == 'gpu':
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    positive_graph = positive_graph.to(torch.device('cuda'))
                    negative_graph = negative_graph.to(torch.device('cuda'))

                input_features = blocks[0].srcdata['feature']
                pos_score, neg_score = self.model(positive_graph=positive_graph,
                                                  negative_graph=negative_graph,
                                                  blocks=blocks,
                                                  x=input_features)
                loss = self.compute_loss(pos_score, neg_score)

                # <---: Back Prop :)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Logging
                results = compute_auc_ap(pos_score, neg_score)

                mlflow.log_metric("Epoch Loss", float(loss.item()), step=step)
                mlflow.log_metric("Model Training AUC",
                                  results['AUC'], step=step)
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

                # Break if number of epochs has been satisfied
                if step == self.params['num_epochs']:
                    break

    def train(self):
        """ Function performs training of the GNN
        and returns the run ID so that the validation and testing set
        accuracy can be recorded against the same run.
        """
        self.make_optimiser()
        self.model.train()
        self.train_epochs()
        return self.model

