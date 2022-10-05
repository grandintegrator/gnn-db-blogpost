# Databricks notebook source
# Library Imports
import argparse
import logging
import yaml
import tqdm 
try:
  import cleanco
except ModuleNotFoundError:
  !pip install cleanco
import cleanco
from warnings import simplefilter
from typing import Dict, Any, List

# Graph machine learning:
!pip install dgl

# Uncomment below if you want to run on GPU
# !pip install -U spacy
# !pip install -U thinc
# !pip install -U pydantic
# !pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
# !pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

import dgl
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
from sklearn.manifold import TSNE
from torch import cat, ones, zeros
from typing import Dict, Any
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch.nn import Sigmoid
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
import pandas as pd
import torch.optim as optim
from torch import cat, ones, zeros
import torch.nn.functional as F

# COMMAND ----------

# MAGIC %md 
# MAGIC Helper functions

# COMMAND ----------

def get_datasets_from_git(data_path: str) -> None:
  import os
  import zipfile
  
  dbutils.fs.mkdirs(f"dbfs:/FileStore/{data_path}/stream_landing_location/")
  working_dir = os.path.split(os.getcwd())[0]
  
  with zipfile.ZipFile(f"{working_dir}/data/streamed_data.zip", "r") as zip_ref:
      zip_ref.extractall(f"/dbfs/FileStore/{data_path}" + "/stream_landing_location")
  
  with zipfile.ZipFile(f"{working_dir}/data/finance_data.zip","r") as zip_ref:
        zip_ref.extractall(f"/dbfs/FileStore/{data_path}")

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

# MAGIC %md Parameters are defined in setup, they are basic and are later tuned using HyperOpt

# COMMAND ----------

params = {
  "test_p": 0.10,
  "valid_p": 0.20,
  "optimiser": "Adam",
  "loss": "binary_cross_entropy",
  "num_node_features": 15,
  "num_hidden_graph_layers": 20,
  "num_negative_samples": 10,
  "num_classes": 2,
  "batch_size": 12, # Mini batch size for the graph
  "num_epochs": 200,
  "num_workers": 0,
  "lr": 0.001,
  "l2_regularisation": 0.0005,
  "momentum": 0.05,
  "aggregator_type": "mean", 
  "device": "cpu"
}

# COMMAND ----------

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

# COMMAND ----------


