# Databricks notebook source
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
try:
  import dgl
except ModuleNotFoundError:
  !pip install dgl
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



# COMMAND ----------

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

# # Load run configuration settings from config file
# with open('../config/config.yaml', 'r') as config_file:
#     params = yaml.safe_load(config_file)

# spark.sql(f"USE {params['database']}")
# logging.info(f"Using {params['database']}")

# COMMAND ----------


