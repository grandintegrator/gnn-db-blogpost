# Databricks notebook source
# MAGIC %md 
# MAGIC ## 2. Some theory on Graph Representation Learning
# MAGIC 
# MAGIC tldr; You may skip to [implementation](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/3734976728300106) if the reader is familiar with GNN theory.
# MAGIC 
# MAGIC ### 2.1 Overview and Motivation
# MAGIC Graphs are a data structure that model a set of objects (nodes or vertices) and their relationships (edges). Graph Neural Networks (GNNs) were born from an exciting field of machine learning research, graph representation learning, which focuses on graph structured data. These methods have achieved state of the art results in chemical synthesis, 3D-vision, recommender systems, question answering, and social network analysis. Prior to the emergence of these algorithms, performing pattern recognition tasks over connected data structures in the field involved some form of filtering and distilling information to derive features. These features would then be used to train simple tabular-based machine learning models or borrow from traditional graph theory. Unfortunately in the distillation process, these methods would lose vital information and lose the rich connectivity patterns which could be exploited for pattern recognition. Additionally, hand engineering features is an inflexible, time consuming, and costly exercise which is vaguely reminiscent of pre-deep learning computer vision.

# COMMAND ----------

# MAGIC %md ### 2.2 Graph Embeddings
# MAGIC In fact, images (and text) can be thought of as graphs with strict connectivity structure. Images can be represented as a grid of pixels (nodes) where the edges represent pixel connectivity. Graphs are therefore a generalisation of both images and text structured data. Borrowing from computer vision, reasoning over graph structured data also involves generating learned embeddings for graphs. The embedding vectors encode structural information about the graphs. There are broadly two methods for generating these embedding vectors: shallow embeddings which make use of an encoder-decoder model and Graph Neural Networks (GNNs). This post focuses on GNNs as they are _inductive_, meaning that the entire set of nodes does not need to be known during training. The converse set of methods are _transductive_ which assume that the full set of nodes are present during the embedding process. For supply chain networks, a key problem constraint is that the network is dynamic due to company dynamics. Therefore the encoding method we choose must be adaptable to changes in the underlying network as new information is gathered. 
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/modern_stack.png" alt="graph-structured-data" width="1000">
# MAGIC <figcaption align="center"><b>Fig.1 Language and images contain implicit structure allowing for convolutions. How do we generalise convolutions across arbitrary structures? </b></figcaption>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Neural Networks
# MAGIC Graph Neural Networks (GNNs) are a general framework for generating embeddings for graph structured data. They employ an entirely new deep learning architecutre that can be thought of as the generalisation of image convolutions (which make use of locality and translational invariance). GNNs generate embeddings by making use of connectivity patterns and feature vectors that live on nodes. Embeddings are generating through neural message passing by effectively defining a computation graph for each node in the graph based on it's local connectivity structure (as shown in Figure 3). For this post we leverage the GraphSAGE architecture for generating our embeddings since.
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/inductive_embeddings.png" alt="embeddings" width="900">

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Reasoning over Embeddings
# MAGIC Within the graph representation learning context, the tasks can be split into **regression and classification over nodes, edges, or the entire graph**. We're going to focus on link prediction which asks what the likelihood of two nodes being connected are given the patterns of connectivity within the graph.
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/latents.png" alt="learning-tasks" style="display: block; margin-left: auto; margin-right: auto; width=900;" width="900"/>

# COMMAND ----------

# MAGIC %md
# MAGIC hello \lamda

# COMMAND ----------

# MAGIC %md
# MAGIC Something here \\( c = \\pm\\sqrt{a^2 + b^2}  \\)
# MAGIC 
# MAGIC \\(A{_i}{_j}=B{_i}{_j}\\)
# MAGIC 
# MAGIC $$c = \\pm\\sqrt{a^2 + b^2}$$
# MAGIC 
# MAGIC \\[A{_i}{_j}=B{_i}{_j}\\]

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Training for Link Prediction
# MAGIC 
# MAGIC Link prediction...
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/graph_training.png" alt="graph-training" width="1000"/>

# COMMAND ----------


