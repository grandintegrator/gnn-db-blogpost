# Databricks notebook source
# MAGIC %md 
# MAGIC # 3. Implementation of our Supply Chain GNN Model

# COMMAND ----------

# MAGIC %md-sandbox ## 3.1 Data Engineering
# MAGIC <div style="float:right">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/data_engineering.png" alt="graph-training" width="840px", />
# MAGIC </div>
# MAGIC 
# MAGIC We begin by ingesting our streaming data using Autoloader and saving as a delta table. Additionally we read in CSV files from our internal teams and convert them to delta tables for more efficient querying.

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

# DBTITLE 1,Defining as streaming source (our streaming landing zone) and destination, a delta table called bronze_company_data
data_location = "dbfs:/FileStore/ajmal_aziz/gnn-blog/data/"

bronze_relation_data = spark.readStream\
                         .format("cloudFiles")\
                         .option("cloudFiles.format", "json")\
                         .option("cloudFiles.maxFilesPerTrigger", 1)\
                         .option("cloudFiles.schemaLocation", data_location+"bronze_schema_location")\
                         .option("cloudFiles.inferColumnTypes", "true")\
                         .load(data_location+"stream_landing_location")

bronze_relation_data.writeStream\
                   .option("checkpointLocation", data_location+"checkpoint_bronze_stream_location") \
                   .trigger(processingTime='1 seconds') \
                   .table("bronze_relation_data")

# COMMAND ----------

# MAGIC %md There are two pieces of data engineering required for this raw Bronze data:
# MAGIC 1. Our ingestion scheme provides a probability and we do not want to train our GNN on low probability links so we will remove below a threshold.
# MAGIC 1. We notice that raw company names have postfixes (e.g Ltd., LLC) denoting different legal entities but pointing to the same physical company. 

# COMMAND ----------

# DBTITLE 1,1. We note that the data set includes low likelihood pairs and should be cut
bronze_company_data = spark.read.table("bronze_relation_data")
draw_probability_distribution(bronze_company_data, probability_col='probability')

# COMMAND ----------

# DBTITLE 1,2. We can use the cleanco package to remove legal entity postscripts within our raw data
print(f"Cleaned example 1: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 LLC')}")
print(f"Cleaned example 2: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 Ltd.')}")

# COMMAND ----------

# DBTITLE 1,We register this logic as a UDF and apply both DE tasks to our collected bronze table
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

clean_company_name = udf(lambda x: cleanco.basename(x), StringType())

silver_relation_data = spark.readStream\
                      .option("cloudFiles.inferColumnTypes", "true")\
                      .table("bronze_relation_data")\
                      .filter(F.col("probability") >= 0.55)\
                      .withColumn("Purchaser", clean_company_name(F.col("Purchaser")))\
                      .withColumn("Seller", clean_company_name(F.col("Purchaser")))

(silver_relation_data.writeStream\
 .format('delta')\
 .option("checkpointLocation", data_location+"/checkpoint_silver_relations")\
 .option('mergeSchema', 'true')\
 .trigger(processingTime="1 seconds")\
 .table('silver_relation_data'))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:right">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/training_GNN.png" alt="graph-training" width="840px", />
# MAGIC </div>
# MAGIC 
# MAGIC ## 3.2 Training our Graph Neural Network
# MAGIC We will train our GNN in the mini-batch setting since this format scales well as the number of nodes in the graph grow, this method of training is also referred to as stochastic training of GNNs. We will leverage a graph data loader class from dgl to facilitate the mini-batch training. Graph data loaders are task dependent. For link prediction, the GNN is trained on **edge batches**. If the task were node focused (node classification/regression) then the graph would be partitioned based on nodes and we would use a node data loader. 
# MAGIC 
# MAGIC Before creating the edge bunches, we partition the graph into a training, validation, and testing graph. For each of the graph splits we negatively sample negative edges (edges that do not exist within the current graph). The negative sampling scheme can get quite complex but we'll maintain a simple negative sampler for this blog. During training batches, a positive graph (actual set of connections) and a negative graph (from our negatively sampled edges) will be used to train the GNN. The GNN loss will incentivise the embeddings to score the likelihood of the negative edges as lower than real edges.

# COMMAND ----------

# DBTITLE 1,We begin by making a DGL (Deep Graph Library) Graph object from our Silver table
silver_relation_table = spark.read.format('delta').table('silver_relation_data')

def make_dgl_graph(relation_table) -> dgl.DGLGraph:
        """
        Make a Graph object within DGL using src_id and dst_id from table encoding graph
        Nodes: Companies with unique src_id or dst_id
        Edges: Company A (with src_id) --> buys_from --> Company B (sith dst_id)
        """
        source_company_id = np.array([x.src_id for x in relation_table.select('src_id').collect()])
        destination_company_id = np.array([x.dst_id for x in relation_table.select('dst_id').collect()])
        return dgl.graph((source_company_id, destination_company_id))

supplier_graph = make_dgl_graph(relation_table=silver_relation_table)

# COMMAND ----------

# DBTITLE 1,We can inspect our graph to see the number of edges and nodes
supplier_graph

# COMMAND ----------

# DBTITLE 1,Split out graph to create graph cross-validation sets (training graph, validation graph, and testing graph)
def make_graph_partitions(graph: dgl.DGLGraph,
                          params: Dict[str, Any]) -> Dict[str, dgl.DGLGraph]:
    """
    Function creates graph partitions consisting of:
    1. training graph: initial tuning of parameters
    2. valid graph: final tuning
    3. testing graph: for reporting final accuracy statistics
    """
    src_ids, dst_ids = graph.edges()

    test_size = int(params['test_p'] * len(src_ids))
    valid_size = int(params['valid_p'] * len(dst_ids))
    train_size = len(src_ids) - valid_size - test_size

    # Source and destinations for training
    src_train, dst_train = src_ids[0:train_size], dst_ids[0:train_size]

    # Source and destinations for validation
    src_valid = src_ids[train_size: valid_size + train_size]
    dst_valid = dst_ids[train_size: valid_size + train_size]

    # Source and destinations for testing
    src_test = src_ids[valid_size + train_size:]
    dst_test = dst_ids[valid_size + train_size:]

    return {'training': dgl.graph((src_train, dst_train)),
            'validation': dgl.graph((src_valid, dst_valid)),
            'testing': dgl.graph((src_test, dst_test))}

# COMMAND ----------

# DBTITLE 1,Create Data Loaders in the form of edges (positive and negative) since we are performing Link Prediction
def get_edge_dataloaders(graph_partitions: Dict[str, dgl.DGLGraph], 
                         params: Dict[str, Any]) -> (Dict[str, dgl.dataloading.EdgeDataLoader],
                                                     Dict[str, dgl.DGLGraph],
                                                     dgl.DGLGraph):
    """
    Returns an edge loader for link prediction for each partition of the graph (training, validation, testing)
    with sampling functionand negative sampling function as well as the graph partitions
    """
    data_loaders = {}
    sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
    
    # Draws params['num_negative_samples'] samples of non-existent edges from the uniform distribution
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(params['num_negative_samples'])

    for split in graph_partitions.keys():
        # Create a feature vector for every node [1 x num_node_features]
        graph_partitions[split].ndata['feature'] = (
            torch.randn(graph_partitions[split].num_nodes(), params['num_node_features'])
        )

        # Create the data loader based on the sampler and negative sampler
        data_loaders[split] = dgl.dataloading.EdgeDataLoader(
            graph_partitions[split],
            graph_partitions[split].edges(form='eid'),
            sampler,
            device=params['device'],
            negative_sampler=negative_sampler,
            batch_size=params['batch_size'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=params['num_workers'])

    return data_loaders

# COMMAND ----------

class DataLoader(object):
    def __init__(self, params, spark):
        self.params = params
        self.company_table = (
            spark.read.format('delta').table('silver_company_data')
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

        return {'training': self.training_graph, 'validation': self.validation_graph,
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

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:right">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/architecture.png" alt="graph-training" width="700px", />
# MAGIC </div>
# MAGIC 
# MAGIC ### 3.2.1 Defining the Graph Neural Network model
# MAGIC Our GNN model will consist of two GraphSAGE layers to generate node embeddings and will be trained using the edge data loaders we have defined above. The embeddings are then fed into a seperate (simple) neural network that will take as inputs the embeddings for source and destination nodes and provide a prediction for the likelihood of a link (binary classification). More formally, the neural network acts as \\( f: (\mathbf{h}_u, \mathbf{h}_v )\rightarrow z{_u}{_v} \\). All of the network weights are trained using a single loss function, either a binary cross entropy loss or a margin loss. During training and validation we collect pseudo-accuracy metrics like the loss and the ROC-AUC and use mlflow to track the metrics during training. Additionally, we use HyperOpt to perform Bayesian Optimisation for selecting the optimal set of parameters.

# COMMAND ----------

# DBTITLE 1,Define a simple Neural Network to classify learned embedding vectors for nodes
class MLPPredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        #                      source and destination nodes
        #          num_node_features     |
        #                    |           |
        self.W = nn.Linear(in_features * 2, 1)
        #                                   |
        #                                   |
        #                                  binary classification
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(cat([h_u, h_v], 1))
        return {'score': torch.sigmoid(score)}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

# COMMAND ----------

# DBTITLE 1,Layers: Defining the GraphSAGE layer for inductive learning
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, aggregator_type):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.n_layers = 2

        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=hid_feats,
                              aggregator_type=aggregator_type)
        self.conv2 = SAGEConv(in_feats=hid_feats, out_feats=out_feats,
            aggregator_type=aggregator_type)

    def forward(self, blocks, inputs):
        # inputs are features of nodes
        h = F.relu(self.conv1(blocks[0], inputs))
        h = self.conv2(blocks[1], h)
        return h

    def inference(self, g, x, batch_size, device='cpu'):
        """
        Based on the graph and feature set, return the embeddings from
        the final layer of the GraphSAGE model.
        """
        # Compute representations layer by layer
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(g.number_of_nodes(), self.hid_feats
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

# COMMAND ----------

# DBTITLE 1,Putting it all together with GraphSAGE to generate embeddings and MLPPredictor to classify links
from utils import GraphSAGE

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes,
                 aggregator_type):
        super().__init__()
        # Define the graph convolution flavour --> GraphSAGE
        self.gcn = GraphSAGE(in_features, hidden_features, out_features, aggregator_type)
        self.pred = MLPPredictor(out_features)

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score

    def get_embeddings(self, g, x, batch_size, device):
        return self.gcn.inference(g, x, batch_size, device)

# COMMAND ----------

# DBTITLE 1,We can now inspect our overall GNN + NN architecture
graph_model = Model(in_features=params['num_node_features'],
                    hidden_features=params['num_hidden_graph_layers'],
                    out_features=params['num_node_features'],
                    num_classes=params['num_classes'],
                    aggregator_type=params['aggregator_type'])

graph_model

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3.2.2 Training the Graph Neural Network

# COMMAND ----------

# DBTITLE 1,We define a trainer class to orchestrate our GNN training
class Trainer(object):
    def __init__(self, params: Dict[str, Any], model: torch.nn.Module,
                 train_data_loader: dgl.dataloading.EdgeDataLoader,
                 validation_data_loader: dgl.dataloading.EdgeDataLoader,
                 training_graph: dgl.DGLGraph):
        self.params = params
        self.model = model
        self.training_graph = training_graph
        self.train_data_loader = train_data_loader
        self.opt = None
        self.sigmoid = nn.Sigmoid()

    def make_optimiser(self):
        # Experiment with 2 seperate optimisers
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
            margin_loss = \
                (neg_score_edge.view(n, -1) - pos_score_edge.view(n, -1) + 1) \
                    .clamp(min=0).mean()
            return margin_loss
        elif self.params['loss'] == 'binary_cross_entropy':
            scores = cat([pos_score_edge, neg_score_edge])
            labels = cat([ones(pos_score_edge.shape[0]),
                          zeros(neg_score_edge.shape[0])])
            scores = scores.view(len(scores), -1).mean(dim=1)
            binary_cross_entropy = F.binary_cross_entropy_with_logits(scores, labels)
            return binary_cross_entropy

    def train_epochs(self):
        self.make_optimiser()
        self.model.train()
        
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
                                                  blocks=blocks, x=input_features)
                loss = self.compute_loss(pos_score, neg_score)

                # 🔙  Back Prop :)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Logging
                results = compute_auc_ap(pos_score, neg_score)
                mlflow.log_metric("Epoch Loss", float(loss.item()), step=step)
                mlflow.log_metric("Model Training AUC", results['AUC'], step=step)
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

                # Break if number of epochs has been satisfied
                if step == self.params['num_epochs']:
                    break

        return {'model': self.model,
                'final_auc': results['AUC'],
                'loss': loss.item()}

# COMMAND ----------

# DBTITLE 1,Define a custom MLFlow model for our GNN (this could be moved elsewhere)
class GNNModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def predict(self, context, model_input):
      """
      Assuming model input is a dataframe containing src_id, dst_id
      """
      graph = make_dgl_graph(model_input)

      # Assign node features to the new graph of size [1 x num_node_features]
      graph.ndata['feature'] = torch.randn(graph_partitions[split].num_nodes(),
                                           params['num_node_features'])
      
      # Draws params['num_negative_samples'] samples of non-existent edges from the uniform distribution
      negative_sampler = dgl.dataloading.negative_sampler.Uniform(params['num_negative_samples'])
      sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
      
      # Create the data loader based on the sampler and negative sampler
      data_loader = dgl.dataloading.EdgeDataLoader(
          graph, graph.edges(form='eid'), sampler, device=params['device'],
          negative_sampler=negative_sampler, batch_size=params['batch_size'],
          shuffle=True, drop_last=False, pin_memory=True, num_workers=params['num_workers'])
      
      with torch.no_grad():
        result = []
        for input_nodes, positive_graph, negative_graph, blocks in data_loader:
          with torch.no_grad():
            input_features = blocks[0].srcdata['feature']
            # 🔜 Forward pass through the network.
            pos_score, neg_score = self.model(positive_graph=positive_graph,
                                   negative_graph=negative_graph,
                                   blocks=blocks,
                                   x=input_features)
          result.append(pos_score)
      return torch.cat([result])


# COMMAND ----------

# DBTITLE 1,Now that we have defined our models, we define a search space for our network and graphs samplers and features
params_hyperopt = {'optimiser': hp.choice('optimiser', ["Adam", "SGD"]),
                   'loss': hp.choice('loss', ['binary_cross_entropy', 'margin']),
                   'num_node_features': hp.choice('num_node_features', [2, 10, 20, 50, 100]),
                   'num_hidden_graph_layers': hp.choice('num_hidden_graph_layers', [5, 15, 100]),
                   'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
                   'num_negative_samples': [10, 50, 100, 400],
                   'num_epochs': hp.choice('num_epochs', [1000, 2000, 8000]),
                   'l2_regularisation': hp.choice('l2_regularisation', [5e-4, 5e-3, 5e-2]),
                   'momentum': hp.choice('momentum', [5e-4, 5e-3, 5e-2]),
                   'lr': hp.choice('lr', [1e-3, 1e-5, 1e-2]),
                   'aggregator_type': hp.choice('aggregator_type', ['mean', 'sum'])}
static_params = {'device': 'cpu',
                 'test_p': 0.1,
                 'valid_p': 0.2,
                 'num_workers': 0,
                 'num_classes': 2}
params_hyperopt = {**params_hyperopt, **static_params}

# COMMAND ----------

# DBTITLE 1,We will use HyperOpt to search the GNN and graph-sampling parameter space
graph_partitions = make_graph_partitions(graph=supplier_graph, params=params)
data_loaders = get_edge_dataloaders(graph_partitions=graph_partitions, params=params)

def train_gnn(params_hyperopt):
    trainer = Trainer(params=params_hyperopt,
                      model=graph_model,
                      train_data_loader=data_loaders['training'],
                      validation_data_loader=data_loaders['validation'],
                      training_graph=graph_partitions['training'])
    training_results = trainer.train_epochs()
    return {'loss': -1*training_results['final_auc'], 'status': STATUS_OK,
            'model': training_results['model'], 'param_hyperopt': params_hyperopt}

def run_gnn_hyperopt(trials=Trials()):
  with mlflow.start_run(run_name="GNN-SUPPLY-CHAIN", nested=True):
    # Log the parameters of the model run
    mlflow.set_tag("link-prediction", "GraphSAGE")
    argmin = fmin(train_gnn, params_hyperopt, algo=tpe.suggest, max_evals=100, trials=trials)
    mlflow.log_params(argmin)

    loss = trials.best_trial['result']['loss']
    mlflow.log_metric("final_auc", loss)
    
    # Log our custom GNN model
    #     mlflow.pytorch.log_model("model", python_model=GNNModelWrapper(trials.best_trial['result']['model']))
    
    # We also log the t-SNE of the node embeddings
    #     model = trials.best_trial['result']['model']
    #     with torch.no_grad():
    #         training_graph = graph_partitions['training']
    #         training_graph_embeddings = (
    #             trained_model.get_embeddings(g=training_graph,
    #                                          x=training_graph.ndata['feature'],
    #                                          batch_size=params['batch_size'],
    #                                          device=params['device'])
    #         )
    #     t_sne_plot = plot_tsne_embeddings(graph_embeddings=training_graph_embeddings,
    #                                       chart_name='training_embeddings',
    #                                       save_fig=True)
    #     mlflow.log_figure(t_sne_plot, 'training_embeddings.png')

# COMMAND ----------

run_gnn_hyperopt(trials = SparkTrials())

# COMMAND ----------


