# Databricks notebook source
# MAGIC %md 
# MAGIC # 3. Implementation: Exploration and DE

# COMMAND ----------

# MAGIC %md-sandbox ## 3.1 Data Engineering
# MAGIC <div style="float:right">
# MAGIC   <img src="https://github.com/grandintegrator/gnn-db-blogpost/blob/main/media/step_1-2.png?raw=True" alt="graph-training" width="840px", />
# MAGIC </div>
# MAGIC 
# MAGIC We begin by ingesting our streaming data using Autoloader and saving as a delta table. Additionally we read in CSV files from our internal teams and convert them to delta tables for more efficient querying.

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

# DBTITLE 1,Create notebook widgets for database name and dataset paths
dbutils.widgets.text(name="database_name", defaultValue="gnn_blog_db", label="Database Name")
dbutils.widgets.text(name="data_path", defaultValue="gnn_blog_data", label="FileStore Path")

# COMMAND ----------

# DBTITLE 1,Unzip data and choose a database for analysis
# Get widget values
data_path = dbutils.widgets.get("data_path")
database_name = dbutils.widgets.get("database_name")

# Extract the zip files in the data directory into dbfs
get_datasets_from_git(data_path=dbutils.widgets.get("data_path"))

# Create a database and use that for our downstream analysis
_ = spark.sql(f"create database if not exists {database_name};")
_ = spark.sql(f"use {database_name};")

# COMMAND ----------

full_data_path = f"dbfs:/FileStore/{data_path}/"

# COMMAND ----------

# DBTITLE 1,Defining as streaming source (our streaming landing zone) and destination, a delta table called bronze_company_data
bronze_relation_data = spark.readStream\
                         .format("cloudFiles")\
                         .option("cloudFiles.format", "json")\
                         .option("cloudFiles.schemaLocation", full_data_path+"_bronze_schema_loc")\
                         .option("cloudFiles.inferColumnTypes", "true")\
                         .load(full_data_path + "stream_landing_location")

bronze_relation_data.writeStream\
                    .format("delta")\
                    .option("mergeSchema", "true")\
                    .option("checkpointLocation", full_data_path + "_checkpoint_bronze_stream")\
                    .trigger(once=True)\
                    .table("bronze_relation_data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 3.1.1 Quickly viewing our Data
# MAGIC Let's print out the tables that we are going to be using throughout this blog. We will use these tables to generate our dashboard to view our supply chain health. The only table used during training of the GNN will be the streamed procurement information, the remaining tables will be used to generate the risk dashboard. Also note that edges in the supply chain graph from _bronze_relation_data_ would be Purchaser (node) \\( \rightarrow  \\) Seller (node). Iteratively building these edge bunches reveals the entire supply chain graph.

# COMMAND ----------

# DBTITLE 1,Bronze collected procurement data 
bronze_company_data = spark.read.table("bronze_relation_data")
display(bronze_company_data)

# COMMAND ----------

# DBTITLE 1,Read our finance tables and register them into our Database
# Read from CSV file and save as Delta
def read_and_write_to_db(table_name: str) -> None:
  (spark.read\
        .option("inferSchema", "true")\
        .option("header", "true")\
        .option("delimiter", ",")\
        .csv(f"dbfs:/FileStore/{data_path}/finance_data/{table_name}.csv")\
        .write.format("delta").mode("overwrite").saveAsTable(table_name))

read_and_write_to_db("company_risk_data")
read_and_write_to_db("company_locations")
read_and_write_to_db("company_risk_frame")

# COMMAND ----------

# DBTITLE 1,Country risk premiums (please note this data is randomised for demo purposes)
display(spark.table("company_risk_data"))

# COMMAND ----------

# DBTITLE 1,Locations of companies
display(spark.table("company_locations"))

# COMMAND ----------

# DBTITLE 1,Risk factors associated to companies (again, these are randomised for demo purposes)
display(spark.table("company_risk_frame"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1.2 Some Quick Data Engineering
# MAGIC There are two pieces of data engineering required for this raw Bronze data:
# MAGIC 1. Our ingestion scheme provides a probability and we do not want to train our GNN on low probability links so we will remove below a threshold.
# MAGIC 1. We notice that raw company names have postfixes (e.g Ltd., LLC) denoting different legal entities but pointing to the same physical company. 

# COMMAND ----------

# DBTITLE 1,1. We note that the data set includes low likelihood pairs which should not be used during training
draw_probability_distribution(bronze_company_data, probability_col='probability')

# COMMAND ----------

# DBTITLE 1,2. We can use the cleanco package to remove legal entity postscripts within our raw data
print(f"Cleaned example 1: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 LLC')}")
print(f"Cleaned example 2: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 Ltd.')}")

# COMMAND ----------

# DBTITLE 1,We register this logic as a UDF and apply both DE tasks to our collected bronze table
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

clean_company_name = udf(lambda x: cleanco.basename(x), StringType())

silver_relation_data = spark.readStream\
                      .format("cloudFiles")\
                      .option("cloudFiles.inferColumnTypes", "true")\
                      .table("bronze_relation_data")\
                      .filter(F.col("probability") >= 0.55)\
                      .withColumn("Purchaser", clean_company_name(F.col("Purchaser")))\
                      .withColumn("Seller", clean_company_name(F.col("Purchaser")))

(silver_relation_data.writeStream\
 .format('delta')\
 .option("checkpointLocation", full_data_path+"_checkpoint_silver_relations")\
 .option('mergeSchema', 'true')\
 .trigger(once=True)\
 .table('silver_relation_data'))

# COMMAND ----------

display(spark.table("silver_relation_data"))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will use this Silver data to train our GNN and refine the low likelihood links that we omitted going from Bronze to Silver. The GNN will be trained on relatively confident links in the next notbook.

# COMMAND ----------


