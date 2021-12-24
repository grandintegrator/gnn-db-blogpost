import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
# from dgl.nn.pytorch.conv import GraphConv


# class StochasticTwoLayerGCN(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
#         self.conv2 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
#
#     def forward(self, blocks, x):
#         x = F.relu(self.conv1(blocks[0], x))
#         x = F.relu(self.conv2(blocks[1], x))
#         return x


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats,
            out_feats=hid_feats,
            aggregator_type='mean'
        )
        self.conv2 = SAGEConv(
            in_feats=hid_feats,
            out_feats=out_feats,
            aggregator_type='mean'
        )

    def forward(self, blocks, inputs):
        # inputs are features of nodes
        h = self.conv1(blocks[0], inputs)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h

#
# class StochasticTwoLayerRGCN(nn.Module):
#     def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
#         super().__init__()
#         self.conv1 = dglnn.HeteroGraphConv({
#                 rel: dglnn.SAGEConv(in_feat, hidden_feat,
#                                     aggregator_type='mean')
#                 for rel in rel_names
#             })
#         self.conv2 = dglnn.HeteroGraphConv({
#                 rel: dglnn.SAGEConv(hidden_feat, out_feat,
#                                     aggregator_type='mean')
#                 for rel in rel_names
#             })
#         self.num_layers = 2
#
#     def forward(self, blocks, x):
#         x = self.conv1(blocks[0], x)
#         # x = dict(map(lambda x: (x[0], F.relu(x[1])), x.items()))
#         x = self.conv2(blocks[1], x)
#         # x = dict(map(lambda x: (x[0], F.leaky_relu(x[1])), x.items()))
#         # x = dict(map(lambda x: (x[0], torch.sigmoid(x[1])), x.items()))
#         return x
#
#     def inference(self, g, x, batch_size, device='cpu'):
#         """
#         Args:
#             g:
#             x:
#             batch_size:
#             device:
#
#         Returns:
#
#         """
#         # Compute representations layer by layer
#         for l, layer in enumerate([self.conv1, self.conv2]):
#             y = torch.zeros(g.number_of_nodes(),
#                             self.hidden_features
#                             if l != self.n_layers - 1
#                             else self.out_features)
#             sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#             dataloader = dgl.dataloading.NodeDataLoader(
#                 g, torch.arange(g.number_of_nodes()), sampler,
#                 batch_size=batch_size,
#                 shuffle=True,
#                 drop_last=False)
#
#             # Within a layer, iterate over nodes in batches
#             for input_nodes, output_nodes, blocks in dataloader:
#                 block = blocks[0]
#
#                 # Copy the features of necessary input nodes to GPU
#                 h = x[input_nodes].to(device)
#                 # Compute output.  Note that this computation is the same
#                 # but only for a single layer.
#                 h_dst = h[:block.number_of_dst_nodes()]
#                 h = F.relu(layer(block, (h, h_dst)))
#                 # Copy to output back to CPU.
#                 y[output_nodes] = h.cpu()
#
#             x = y
