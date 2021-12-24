# Databricks notebook source
# MAGIC %md
# MAGIC # Supply Chain Crisis? Monitoring Supply Chains in real-time with Databricks and Graph Neural Networks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Supply Chains as Networks
# MAGIC We live in an ever interconnected world. Nowhere is this more evident than our global supply chains. Modern supply chains have been intricately linked and weaved together due to the glocal macroeconomic environment and globalisation. In modern supply chains, companies rely on one another to keep their production lines flowing and even to act in an ethical manner by sourcing from ethical suppliers. The relationships between firms in this global network of companies buying and selling from another form a dynamic complex network. Most companies typically consider only a single set of connections in this complex network, or their tier 1 suppliers exposing themselves to significant amounts of risk. 
# MAGIC 
# MAGIC This post focuses on building resilient supply chains using the power of data and machine learning by showing how companies can have a real-time view of their supply chain. We will develop a dashboard that executives or supply chain analysts can use to monitor their supply chain. We will assume for this post that a company has developed a way to gather this dataset either by using commercial databases or web scraping ([see here for proposals on how to do this](google.com)). Since the supply chain network is inevitably going to be incomplete and change over time, we will also demonstrate training and productionising Graph Neural Networks (GNNs) to continuously monitor and refine the supply chain network. The GNN will learn from connectivity patterns that have been gathered to find missing information in the form of links. Formally, in the graph representation learning literature, this prediction task is described as link prediction.

# COMMAND ----------

# MAGIC %md 
# MAGIC <img align="middle" src="files/ajmal_aziz/gnn-blog/supplier_network.png"  alt="graph-structured-data">
# MAGIC <figcaption align="center"><b>Fig.1 Arrows between suppliers represent buying and selling relationships. Firms typically see one layer of this network whilst there a web of other suppliers (and some potentially high risk) that are invisible. </b></figcaption>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resilient Supply Chains
# MAGIC The first step to building resilient supply chains is to first have visbility over the supply chain itself. There have been multiple instances where companies have fallen victim to a lack of visibility in their supply chains. This has led to both financial and ESG implications for these firms.

# COMMAND ----------


