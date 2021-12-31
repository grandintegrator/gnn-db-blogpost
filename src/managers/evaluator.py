import torch
import tqdm
from utils import compute_auc_ap
import mlflow.pytorch
from typing import List


class Evaluator(object):
    def __init__(self, params, model, testing_data_loader):
        self.params = params
        self.model = model
        self.testing_data_loader = testing_data_loader
        self.sigmoid = torch.nn.Sigmoid()

    def evaluate(self) -> (List, List):
        """Function evaluates the model on all batches and saves all values
        Returns:
        """
        # Turn model into evaluation mode
        self.model.eval()
        roc_auc_all = []
        ap_all = []
        with tqdm.tqdm(self.testing_data_loader) as tq:
            for step, (input_nodes, positive_graph, negative_graph,
                       blocks) in enumerate(tq):
                with torch.no_grad():
                    input_features = blocks[0].srcdata['feature']
                    # 🔜 Forward pass through the network.
                    pos_score, neg_score = \
                        self.model(positive_graph=positive_graph,
                                   negative_graph=negative_graph,
                                   blocks=blocks,
                                   x=input_features)

                    auc_pr_dicts = compute_auc_ap(pos_score, neg_score)
                    roc_auc_all.append(auc_pr_dicts['AUC'])
                    ap_all.append(auc_pr_dicts['AP'])

            mlflow.log_param('validation_mean_roc', sum(roc_auc_all)/len(roc_auc_all))
            mlflow.log_param('validation_mean_ap', sum(ap_all)/len(ap_all))

        return roc_auc_all, ap_all
