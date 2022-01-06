# Databricks notebook source
# MAGIC %md 
# MAGIC # LIBRARY IMPORTS - PLEASE IGNORE, VERY UGLY, WILL CLEAN LATER I PROMISE!!

# COMMAND ----------

import argparse
import logging
import yaml
import tqdm 
try:
  import cleanco
except ModuleNotFoundError:
  !pip install cleanco
from warnings import simplefilter
from typing import Dict, Any, List

import numpy as np
import torch
import mlflow
import mlflow.pytorch
import hyperopt
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from hyperopt import space_eval
from pyspark.sql.functions import lit
from mlflow.models.signature import infer_signature
from utils import compute_auc_ap, plot_tsne_embeddings

# Graph machine learning:
try:
  import dgl
except ModuleNotFoundError:
  !pip install dgl
  
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
import pandas as pd
import torch.optim as optim
from torch import cat, ones, zeros
import torch.nn.functional as F

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


# COMMAND ----------

# import mlflow
# class GNNModelWrapper(mlflow.pyfunc.PythonModel):
#     def __init__(self, model, params):
#         self.model = model
#         self.params = params
    
#     def predict(self, context, model_input):
#       """
#       Assuming model input is a dataframe containing src_id, dst_id
#       """
#       graph = make_dgl_graph(model_input)

#       # Assign node features to the new graph of size [1 x num_node_features]
#       graph.ndata['feature'] = torch.randn(graph_partitions[split].num_nodes(),
#                                            params['num_node_features'])
    
#       # Draws params['num_negative_samples'] samples of non-existent edges from the uniform distribution
#       negative_sampler = dgl.dataloading.negative_sampler.Uniform(params['num_negative_samples'])
#       sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
      
#       # Create the data loader based on the sampler and negative sampler
#       data_loader = dgl.dataloading.EdgeDataLoader(
#           graph, graph.edges(form='eid'), sampler, device=params['device'],
#           negative_sampler=negative_sampler, batch_size=params['batch_size'],
#           shuffle=True, drop_last=False, pin_memory=True, num_workers=params['num_workers'])
      
#       with torch.no_grad():
#         result = []
#         for input_nodes, positive_graph, negative_graph, blocks in data_loader:
#           with torch.no_grad():
#             input_features = blocks[0].srcdata['feature']
#             # 🔜 Forward pass through the network.
#             pos_score, neg_score = self.model(positive_graph=positive_graph,
#                                    negative_graph=negative_graph,
#                                    blocks=blocks,
#                                    x=input_features)
#           result.append(pos_score)
#       return torch.cat([result])


# COMMAND ----------

def draw_probability_distribution(dataframe, probability_col):
  dataframe = dataframe.toPandas()
  dataframe[probability_col] = dataframe[probability_col].astype('float')
  
  import plotly.express as px
  import plotly.io as pio
  pio.templates.default = "plotly_white"
  
  fig = px.violin(dataframe,
                  x=probability_col,
                  box=True, # draw box plot inside the violin
                  points='all', # can be 'outliers', or False
                 )
  fig.update_layout(font_family='Arial',
                  title='Distribution of confidence values',
                  yaxis_title=r"Density",
                  xaxis_title=r"Probability",
                  font=dict(size=24))
  fig.show()

# COMMAND ----------

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

# Load run configuration settings from config file
with open('../config/config.yaml', 'r') as config_file:
    params = yaml.safe_load(config_file)

spark.sql(f"USE {params['database']}")
logging.info(f"Using {params['database']}")

# COMMAND ----------


