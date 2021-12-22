# Databricks notebook source
# MAGIC %md 
# MAGIC ## 2. Some theory on Graph Representation Learning
# MAGIC 
# MAGIC tldr; You may skip to [implementation](https://www.youtube.com/watch?v=dQw4w9WgXcQ) if the reader is familiar with GNN theory.
# MAGIC 
# MAGIC ### Overview
# MAGIC Graph Neural Networks (GNNs) are an exciting new field of machine learning research which address graph structured data. Historically performing pattern recognition tasks over connected data required filtering and deriving features from networks to be fed into machine learning models. These vectorised features would distill information and lose the rich connectivity patterns that could be useful for patterns which could be exploited. Thus, these new algorithms that naturally model data as graphs 
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/modern_stack.png" alt="graph-structured-data" width="1000">
# MAGIC <figcaption align="center"><b>Fig.1 Language and images contain implicit structure allowing for convolutions. How do we generalise convolutions across arbitrary structures? </b></figcaption>
# MAGIC 
# MAGIC 
# MAGIC ### Generating Embeddings
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/graph_embedding_transductive.png" alt="embeddings" width="900">
# MAGIC 
# MAGIC ### Reasoning over Embeddings
# MAGIC Graph structured data is not independent and identically distributed (iid) as nodes and edges inherently model dependency structures. Reasoning over these structures rely on embedding vectors. Within the graph representation learning context, the tasks can be split into **regression and classification over nodes, edges, or the entire graph**. We're going to focus on link prediction which asks what the likelihood of two nodes being connected are given the patterns of connectivity within the graph.
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/latents.png" alt="learning-tasks" style="display: block; margin-left: auto; margin-right: auto; width=900;" width="900"/>

# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC ### 3. Training Graph Neural Networks
# MAGIC <img src="files/ajmal_aziz/gnn-blog/graph_training.png" alt="graph-training" width="1000"/>

# COMMAND ----------


