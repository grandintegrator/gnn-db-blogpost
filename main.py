import argparse
import logging
import pprint
import yaml
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pathlib import Path
from warnings import simplefilter


def main(run_args) -> None:
    """
    Main entry point for the project
    """
    simplefilter(action='ignore', category=UserWarning)
    spark = SparkSession.builder.getOrCreate()
    spark.sql("""USE ajmal_aziz_gnn_blog""")
    spark.sql("""select * from country_risk""")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='GNN Blog Post Run Arguments.')

    # Parser arguments for run...
    parser.add_argument("--notebook", type=bool, default='True',
                        help="Is this being run from inside Databricks?")

    run_args = parser.parse_args()
    main(run_args)
