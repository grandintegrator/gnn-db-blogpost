import argparse
import logging
import yaml
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pathlib import Path
from warnings import simplefilter
from dataset.dataloader import DataLoader
from managers.trainer import Trainer
from managers.evaluator import Evaluator
from utils import create_model, plot_tsne_embeddings
import torch
import mlflow.pytorch


def main(config) -> None:
    """
    Main entry point for the project
    """
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=FutureWarning)
    spark = SparkSession.builder.getOrCreate()

    # Load run configuration settings from config file
    with open('config/config.yaml', 'r') as config_file:
        params = yaml.safe_load(config_file)

    # --------------------------------------------------------------------------
    # Spark parameters and configuration
    # --------------------------------------------------------------------------
    spark.sql(f"USE {params['database']}")
    logging.info(f"Using {params['database']}")

    # --------------------------------------------------------------------------
    # Create dataset as well as dataloaders for training, validation and testing
    # --------------------------------------------------------------------------
    loader = DataLoader(params=params, spark=spark)
    data_loaders, graph_partitions, _ = loader.get_edge_dataloaders()

    # --------------------------------------------------------------------------
    # Create a graph model for training
    # --------------------------------------------------------------------------
    graph_model = create_model(params=params)

    # --------------------------------------------------------------------------
    # Start mlflow training run
    # --------------------------------------------------------------------------
    with mlflow.start_run(run_name='GNN-BLOG-MODEL') as run:
        # Log the parameters of the model run
        mlflow.log_params(params)
        mlflow.set_tag("link-prediction", "graphSAGE")

        # Train the GNN encoder along with the MLP link predictor
        logging.info('Starting training....')
        trainer = Trainer(params=params,
                          model=graph_model,
                          train_data_loader=data_loaders['training'],
                          validation_data_loader=data_loaders['validation'],
                          training_graph=graph_partitions['training'])
        trained_model = trainer.train()

        # Log the model artefacts and parameters for the run
        trained_model.eval()
        with torch.no_grad():
            training_graph = graph_partitions['training']
            training_graph_embeddings = (
                trained_model.get_embeddings(g=training_graph,
                                             x=training_graph.ndata['feature'],
                                             batch_size=params['batch_size'],
                                             device=params['device'])
            )
        plot_tsne_embeddings(graph_embeddings=training_graph_embeddings,
                             chart_name='training_embeddings',
                             save_fig=True)
        mlflow.log_artifact('data/training_embeddings.png')

        # --------------------------------------------------------------------------
        # Evaluate model accuracy
        # --------------------------------------------------------------------------
        evaluator = Evaluator(params=params,
                              model=trained_model,
                              testing_data_loader=data_loaders['validation'])
        auc_list, ap_list = evaluator.evaluate()


        # mlflow.pytorch.log_model(self.model, "model")
        # summary, _ = count_model_parameters(net)
        # mlflow.log_text(str(summary), "model_summary.txt")
        # mlflow.set_tracking_uri("http://localhost:5000")
        # mlflow.pytorch.log_model(
        #     pytorch_model=self.model,
        #     artifact_path='mlruns:/0/{}/model'.format(run.info.run_id),
        #     registered_model_name="GNN-BLOG-MODEL"
        # )
        # logging.info("Training and logging complete")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("py4j").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='GNN Blog Post Run Arguments.')

    # Parser arguments for run...
    parser.add_argument("--notebook", type=bool, default='True',
                        help="Is this being run from inside Databricks?")

    run_args = parser.parse_args()
    main(run_args)
