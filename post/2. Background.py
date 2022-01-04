# Databricks notebook source
# MAGIC %md 
# MAGIC ## 2. Some Background
# MAGIC 
# MAGIC tldr; You may skip to [implementation](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/3734976728300106) if the reader is familiar with GNN theory. This section focuses on _just enough_ theory to understand the implementation section.
# MAGIC 
# MAGIC ### 2.1 Overview and Motivation
# MAGIC Graphs are a data structure that model a set of objects (nodes or vertices) and their relationships (edges). Graph Neural Networks (GNNs) were born from an exciting field of machine learning research, graph representation learning, which focuses on graph structured data. These methods have achieved state of the art results in chemical synthesis, 3D-vision, recommender systems, question answering, and social network analysis just to name a few. Prior to the emergence of these algorithms, performing pattern recognition tasks over connected data structures in the field involved some form of filtering and distilling information to derive predictors or _features_. These features would then be used to train simple tabular-based machine learning models or borrow from traditional graph theory. Unfortunately during the _distillation_ process, these methods would lose vital information and lose the rich connectivity patterns which could be exploited for pattern recognition. Additionally, hand engineering features is an inflexible, time consuming, and costly exercise which is vaguely reminiscent of pre-deep learning computer vision.

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC ### 2.2 Graph Embeddings
# MAGIC In fact, images (and text) can be thought of as graphs with strict connectivity structure. Images can be represented as a grid of pixels (nodes) where the edges represent pixel connectivity. Graphs are therefore a generalisation of both images and text structured data. Borrowing from computer vision, reasoning over graph structured data in the deep learning context also involves generating learned embeddings from graphs. The embedding vectors encode structural information about the graphs which can exploited in downstream tasks. There are broadly two methods for generating these embedding vectors: shallow embeddings which make use of an encoder-decoder model and Graph Neural Networks (GNNs), a more complex encoding scheme. This post focuses on GNNs as they can be _inductive_, meaning that the entire set of nodes in the graph do not need to be known during training which suits our use case. The converse set of methods are _transductive_ which assume that the full set of nodes are present during the embedding process. For supply chain networks, a key problem constraint is that the network is dynamic due to company dynamics. Therefore the encoding method we choose must be adaptable to changes in the underlying network as new information is gathered. 
# MAGIC 
# MAGIC <div style="float: center">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/text_image_network.png" alt="graph-structured-data" width="1200px">
# MAGIC   <figcaption align="center"><b>Fig.1 Language and images contain implicit structure allowing for convolutions. How do we generalise convolutions across arbitrary structures? </b></figcaption>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Reasoning over Embeddings
# MAGIC Once embeddings have been obtained for a graph, downstream tasks can be split into **regression and classification over nodes, edges, or the entire graph** depending on the problem specified. Note here that reasoning over graph structured data is more expressive than traditional machine learning techniques focused on matrix, text, or image data. Within graphs, we could be interested in the nodes, the edges or the entire graph. Some examples of learning tasks include:
# MAGIC 
# MAGIC | Learning Task                         | Problem Statement                                                                               | Graph            | Nodes    | Edges                    |
# MAGIC |---------------------------------------|-------------------------------------------------------------------------------------------------|------------------|----------|--------------------------|
# MAGIC | **Graph Classification**                  | Classify whether or not a given protein structure for a drug will inhibit a bacterial infection | Protein Network  | Proteins | Bonds                    |
# MAGIC | **Node Classification**                   | Identify whether or not a scientific paper is about diabetes                                    | Citation Network | Papers   | Citations between papers |
# MAGIC | **Link Prediction (edge classification)** | Are two people friends?                                                                         | Social Network   | People   | Known friendships        |
# MAGIC 
# MAGIC <br> 
# MAGIC Let us bring some math into the fold. Assuming we have a graph \\( \mathcal{G} = (\mathcal{V}, \mathcal{E})\\) where \\( \mathcal{V} \\) is the set of nodes in the graph, and \\( \mathcal{E} \\) is the set of edges defined as tuples \\( (u, v) \\) which denode connections between nodes \\( u \\) and \\( v \\) within our graph. Within this context, a GNN is a function that maps \\( u \\), some node in our graph into an embedding vector \\( \mathbf{h} \\). For link prediction, and in particular, predicting the existence of a link over our supply chain network, we make use of function that maps two given node embeddings into a range (0, 1) for the likelihood of a link existing between the two nodes. This function is defined as: \\( z_{i, j} = f(\mathbf{h}_i, \mathbf{h}_j)  \\) where \\( \mathbf{h}_i \\) and \\( \mathbf{h}_j \\) denote the embeddings for nodes \\( i \\) and \\( j \\) within our graph. This function is usually defined as \\( \sigma(\mathbf{h}_i \cdot \mathbf{h_j})\\) where \\( \sigma \\) is the sigmoid function. An alternative method is to define a fully connected neural network layer which has it's weights trained during the overall GNN training stage. The next section will focus on how we generate these embeddings in a scalable fashion using a GNN.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Neural Networks
# MAGIC Graph Neural Networks (GNNs) are a framework for generating embeddings for graph structured data. They employ an entirely new deep learning architecutre that can be thought of as the generalisation of image convolutions (which make use of locality and translational invariance) with the added bonus of certain mathematical properties that are necessary to operate over graph structured data. GNNs can generate embeddings by making use of connectivity patterns as well as feature vectors that live on nodes. Embeddings are generated by defining a computation graph for each node in the graph based on it's local connectivity structure (as shown in Figure 3).
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/graph_convolutions.png" alt="embeddings" width="1500">
# MAGIC 
# MAGIC For this post we leverage the GraphSAGE architecture for generating our embeddings since this algorithm can scalably and inductively generate embeddings for new nodes in an efficient manner and scale to billions of nodes. This is useful for streaming workloads where new nodes are introduced and we would like to generate embedding vectors without significant computation costs associated to retraining or extra gradient steps as we would have to in transductive methods. For more information on GraphSAGE, see [here](https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html).

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Training for Link Prediction
# MAGIC 
# MAGIC Link prediction...
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/training_GNN.png" alt="graph-training" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC [1] https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html
