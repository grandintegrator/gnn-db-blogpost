# Databricks notebook source
# MAGIC %md
# MAGIC # Supply Chain Crisis? Monitoring Supply Chains in real-time with Databricks and Graph Neural Networks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Supply Chains as Networks
# MAGIC We live in an ever interconnected world. Nowhere is this more evident than our global supply chains. Modern supply chains have become intricately linked and weaved together due to the global macroeconomic environment and globalisation. In modern supply chains, companies rely on one another to keep their production lines flowing and to act in an ethical manner by ensuring they source from ethical suppliers. The relationships between firms in this global network of companies buying and selling from another form a dynamic complex network.  
# MAGIC 
# MAGIC Most companies typically consider only a single set of connections from this complex network, or their tier 1 suppliers. This exposes firms to significant amounts of risk due to a lack of transparency. This post focuses on building resilient supply chains using the power of data and machine learning by showing how companies can have a real-time view of their supply chain after having collected some representative data. We will develop a dashboard that executives or supply chain analysts can use to monitor their supply chain. We will assume for this post that a company has developed a way to gather this dataset either by using commercial databases or published web information ([see here for automated method using natural language processing and deep learning using web data](https://www.researchgate.net/publication/339103487_Extracting_supply_chain_maps_from_news_articles_using_deep_neural_networks)). 
# MAGIC 
# MAGIC Since the supply chain network is inevitably going to be incomplete and change over time, we will also demonstrate training and productionising Graph Neural Networks (GNNs) to continuously monitor and refine the collected supply chain network data. The GNN will learn from connectivity patterns that have been gathered to find missing information in the form of links. Formally, in the graph representation learning literature, this prediction task is described as link prediction.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="text-align:center">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/supplier_network.png"  alt="graph-structured-data">
# MAGIC </div>
# MAGIC 
# MAGIC <figcaption align="center"><b>Fig.1 Arrows between suppliers represent buying and selling relationships. Firms typically see one layer of this network whilst there a web of other suppliers (and some potentially high risk) that are invisible. </b></figcaption>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2. Building Resilient Supply Chains
# MAGIC The first step to building resilient supply chains is to first have visbility over the supply chain itself. There have been multiple instances where companies have fallen victim to a lack of visibility in their supply chains. Examples include suppliers of suppliers that do not comply with the Modern Slavery Act or are involved in known nefarious activities. Furthermore, there could be production risk associated to suppliers of suppliers from untrustworthy business practices or even hidden geographic risk exposure that. These shocks have led to both financial and reputational implications for firms that have fallen victim to this lack of transparency. 
# MAGIC 
# MAGIC For this post we're going assume that a company has set up the pre-requisite architectre to obtain potential buying and selling relationships between firms with some probability score. This data will be streamed to a cloud storage and will be streamed using ```autoloader```. We then also incoporate data from our supply and finance teams that have structured information about country risk profiles as well as company risk scores. These will be stored as CSV files and ingested in as silver tables without our medallion architecture.
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC <div style="text-align:center">
# MAGIC   <img src="files/ajmal_aziz/gnn-blog/architecture_high_level.png" width=1500px alt="graph-structured-data">
# MAGIC </div>
# MAGIC 
# MAGIC <figcaption align="center"><b>Fig.2 Adopted architecture for this blog post. </b></figcaption>
# MAGIC 
# MAGIC Let's perform some very simple exploratory data analysis to see what the raw data for such systems typically look like

# COMMAND ----------

# DBTITLE 1,Let us quickly explore the raw JSON data that is being collected for our supply chain network
# MAGIC %sql 
# MAGIC select * from json.`dbfs:/FileStore/ajmal_aziz/gnn-blog/data/stream_landing_location`

# COMMAND ----------

# DBTITLE 1,Reading a CSV file from our finance team on country risk information
country_risk_data = spark.read\
                         .option("inferSchema", "true")\
                         .option("header", "true")\
                         .option("delimiter", ",")\
                         .csv("dbfs:/FileStore/ajmal_aziz/gnn-blog/data/country_risk_data.csv")
display(country_risk_data)

# COMMAND ----------

# DBTITLE 1,For known companies, our risk analysts have provides metrics to quantify risks
company_risk_frame = spark.read\
                         .option("inferSchema", "true")\
                         .option("header", "true")\
                         .option("delimiter", ",")\
                         .csv("dbfs:/FileStore/ajmal_aziz/gnn-blog/data/company_risk_frame.csv")
display(company_risk_frame)

# COMMAND ----------

# DBTITLE 1,Lastly we have employed analysts to determine the location of companies for us
company_locations = spark.read\
                         .option("inferSchema", "true")\
                         .option("header", "true")\
                         .option("delimiter", ",")\
                         .csv("dbfs:/FileStore/ajmal_aziz/gnn-blog/data/company_locations.csv")
display(company_locations)
