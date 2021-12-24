import dgl
import logging
from model.dgl.StochasticGCN import Model


def create_model(params):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # in_features, hidden_features, out_features, num_classes,
    # etypes):

    model = Model(in_features=params['num_node_features'],
                  hidden_features=params['num_hidden_graph_layers'],
                  out_features=params['num_node_features'],
                  num_classes=params['num_classes'])
    return model
