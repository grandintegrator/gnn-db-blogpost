import argparse
import logging
import yaml
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pathlib import Path
from warnings import simplefilter
from dataset.dataloader import DataLoader
from managers.trainer import Trainer
# from managers.evaluator import Evaluator
from utils import create_model


def main(config) -> None:
    """
    Main entry point for the project
    """
    simplefilter(action='ignore', category=UserWarning)
    spark = SparkSession.builder.getOrCreate()

    # Load run configuration settings from config file
    with open('config/config.yaml', 'r') as config_file:
        params = yaml.safe_load(config_file)

    # ------------------------------------------------------------------------------------------------------------------
    # Spark parameters and configuration
    # ------------------------------------------------------------------------------------------------------------------
    spark.sql(f"USE {params['database']}")
    logging.info(f"Using {params['database']}")

    # ------------------------------------------------------------------------------------------------------------------
    # Create dataset as well as data loaders for training, validation and testing
    # ------------------------------------------------------------------------------------------------------------------
    loader = DataLoader(params=params, spark=spark)
    data_loaders = loader.get_edge_dataloaders()
    logging.info("Initialised edge data loaders for training, validation, and testing")

    # ------------------------------------------------------------------------------------------------------------------
    # Create a graph model for training
    # ------------------------------------------------------------------------------------------------------------------
    graph_model = create_model(params=params)

    logging.info('Starting training')
    trainer = Trainer(params=params,
                      model=graph_model,
                      train_data_loader=data_loaders['training'])

    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("py4j").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='GNN Blog Post Run Arguments.')

    # Parser arguments for run...
    parser.add_argument("--notebook", type=bool, default='True',
                        help="Is this being run from inside Databricks?")

    run_args = parser.parse_args()
    main(run_args)
