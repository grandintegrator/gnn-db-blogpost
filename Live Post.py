# Databricks notebook source
# MAGIC %md
# MAGIC # Monitoring Supply Chains in real-time with Databricks and Graph Neural Networks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Resilient Supply Chains
# MAGIC We live in an ever interconnected world. Nowhere is this more evident than our global supply chains. Modern supply chains have been intricately linked and weaved together due to the glocal macroeconomic environment. In modern supply chains, companies rely on one another to keep their production lines flowing and even to act in an ethical manner by sourcing from ethical suppliers. Since the complex interconnections of companies buying and selling from another form a complex network, it is difficult for companies to have visibility over their supply chains
# MAGIC 
# MAGIC Unfortunately, most companies don't have visibility over their extended supply chain networks.  This post focuses on building resilient supply chains using the power of data and machine learning. We will develop a dashboard to monitor our supply chain as well as training and productionising using Graph Neural Networks to continuously monitor and refine our network.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Some theory on Graph Neural Networks
# MAGIC 
# MAGIC tldr; You may skip to [implementation](https://www.youtube.com/watch?v=dQw4w9WgXcQ) if the reader is familiar with GNN theory.
# MAGIC 
# MAGIC ### Overview
# MAGIC Graph Neural Networks (GNNs) are an exciting new field of machine learning research which address graph structured data. Historically performing pattern recognition tasks over connected data required filtering and deriving features from networks to be fed into machine learning models. These vectorised features would distill information and lose the rich connectivity patterns that could be useful for patterns which could be exploited. Thus, these new algorithms that naturally model data as graphs 
# MAGIC 
# MAGIC <img src="files/ajmal_aziz/gnn-blog/modern_stack.png" alt="graph-structured-data" width="1000">
# MAGIC <figcaption align="center"><b>Fig.1 Although </b></figcaption>
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

# MAGIC %md
# MAGIC ## 4. Our Solution Architecture

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Implementation

# COMMAND ----------

# MAGIC %md 
