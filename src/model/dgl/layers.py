import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
# from dgl.nn.pytorch.conv import GraphConv


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
