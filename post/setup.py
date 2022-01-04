# Databricks notebook source
import argparse
import logging
import yaml
import tqdm 
try:
  import cleanco
except ModuleNotFoundError:
  !pip install cleanco
from warnings import simplefilter
from typing import Dict, Any

import numpy as np
import torch
import mlflow
import mlflow.pytorch

from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from hyperopt import space_eval
from pyspark.sql.functions import lit

# Graph machine learning:
try:
  import dgl
except ModuleNotFoundError:
  !pip install dgl

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


# COMMAND ----------

from utils import compute_auc_ap, plot_tsne_embeddings

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

import torch.optim as optim
from torch import cat, ones, zeros
import torch.nn.functional as F

# COMMAND ----------


