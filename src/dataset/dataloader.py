import numpy as np
import dgl
import torch
from typing import Dict


class DataLoader(object):
    def __init__(self, params, spark):
        self.params = params
        self.company_table = (
            spark.read.format('delta').table('buys_relations_table')
        )

        self.graph = None
        self.training_graph = None
        self.testing_graph = None
        self.validation_graph = None

    def make_dgl_graph(self) -> dgl.DGLGraph:
        """
        Make a Graph object within DGL using just src_id and dst_id
        """
        source_company_id = \
            np.array([x.src_id for x in self.company_table.select('src_id').collect()])
        destination_company_id = \
            np.array([x.dst_id for x in self.company_table.select('dst_id').collect()])
        return dgl.graph((source_company_id, destination_company_id))

    def make_graph_partitions(self) -> Dict[str, dgl.DGLGraph]:
        """
        Function creates graph partitions consisting of:
         1. training graph: initial tuning of parameters
         2. valid graph: final tuning
         3. testing graph: for reporting final accuracy statistics
        """
        self.graph = self.make_dgl_graph()

        src_ids, dst_ids = self.graph.edges()
        test_size = int(self.params['test_p'] * len(src_ids))
        valid_size = int(self.params['valid_p'] * len(dst_ids))
        train_size = len(src_ids) - valid_size - test_size

        # Source and destinations for training
        src_train = src_ids[0:train_size]
        dst_train = dst_ids[0:train_size]
        # Source and destinations for validation
        src_valid = src_ids[train_size: valid_size + train_size]
        dst_valid = dst_ids[train_size: valid_size + train_size]

        # Source and destinations for testing
        src_test = src_ids[valid_size + train_size:]
        dst_test = dst_ids[valid_size + train_size:]

        self.training_graph = dgl.graph((src_train, dst_train))
        self.validation_graph = dgl.graph((src_valid, dst_valid))
        self.testing_graph = dgl.graph((src_test, dst_test))

        return {'training': self.training_graph,
                'validation': self.validation_graph,
                'testing': self.testing_graph}

    def get_edge_dataloaders(self) -> (Dict[str, dgl.dataloading.EdgeDataLoader],
                                       Dict[str, dgl.DGLGraph],
                                       dgl.DGLGraph):
        """
        Returns an edge loader for link prediction with sampling function
         and negative sampling function as well as the graph partitions
        """
        graph_partitions = self.make_graph_partitions()

        data_loaders = {}

        # TODO: Have the neighbourhood and negative sampler as a parameter that
        # get set during the tuning process
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(10)

        for split in graph_partitions.keys():

            # Associate node features with the graphs
            graph_partitions[split].ndata['feature'] = (
                torch.randn(graph_partitions[split].num_nodes(),
                            self.params['num_node_features'])
            )

            # Create the data loader based on the sampler and negative sampler
            data_loaders[split] = dgl.dataloading.EdgeDataLoader(
                graph_partitions[split],
                graph_partitions[split].edges(form='eid'),
                sampler,
                device=self.params['device'],
                negative_sampler=negative_sampler,
                batch_size=self.params['batch_size'],
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=self.params['num_workers'])

        return data_loaders, graph_partitions, self.graph




