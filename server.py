from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import  test

import flwr as fl

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation accuracy using weighted average."""
        if not results:
            return None, {}
        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["MSE_error"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        # total_writer.add_scalar('total loss', aggregated_accuracy)
        print(f"Round {server_round} MSE aggregated from client results: {aggregated_accuracy}")
        test.e +=1
        print(test.e)
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"MSE_error": aggregated_accuracy}

# Create strategy and run server
strategy = AggregateCustomMetricStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Never sample less than 10 clients for training
    min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
    min_available_clients=4,  # Wait until all 10 clients are available
)
# fl.server.start_server(strategy=strategy)

# # Create simple FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
#     min_fit_clients=6,  # Never sample less than 10 clients for training
#     min_evaluate_clients=6,  # Never sample less than 5 clients for evaluation
#     min_available_clients=6,  # Wait until all 10 clients are available
# )



# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
# client_resources = None
# if DEVICE.type == "cuda":
#     client_resources = {"num_gpus": 1}


