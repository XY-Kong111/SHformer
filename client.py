from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch_model
from torch.utils.tensorboard import SummaryWriter
import flwr as fl
import  test

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
NUM_CLIENTS = 10


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_trans_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_trans_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def epoch_set(epoch):
    if epoch == 0 or None:
        epoch = 1
    else:
        epoch = epoch+1
    return epoch
def global_lr(epoch):
    lr = 0.001
    learning_rate = lr * (0.95 ** ((epoch - 1) // 1))
    return learning_rate


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, testloader, name):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.dir = f'log/CO2_emission_acconting/federated/{name}/{cid}'
        self.writer = SummaryWriter(self.dir)
        self.epoch = epoch_set(test.e)
        self.learning_rate = global_lr(self.epoch)
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)


    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, learning_rate  : {self.learning_rate}")
        set_parameters(self.net, parameters)
        torch_model.train(self.net, self.trainloader,self.valloader, 1, self.writer, self.learning_rate)
        # self.epoch += 1
        # self.learning_rate  = self.learning_rate  * (0.95 ** ((self.epoch - 1) // 1))
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss = torch_model.test(self.net, self.testloader, self.writer, self.dir,  self.epoch)
        return float(loss), len(self.testloader), {"MSE_error": float(loss)}


class FlowerNumPyClientTrans(fl.client.NumPyClient):
    def __init__(self, cid, exp,model, trainloader, valloader, testloader, setting):
        self.setting = setting
        self.cid = cid
        self.exp = exp
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.writer = SummaryWriter('log/CO2_emission_acconting/federated/%s'%(cid))

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.model, parameters)
        self.exp.train(self.trainloader,self.valloader,self.testloader,self.setting )
        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.model, parameters)
        rmse = self.exp.test(self.testloader, self.setting)
        return float(rmse), len(self.testloader), {"MSE_error": float(rmse)}

# def numpyclient_fn(cid) -> FlowerClient:
#     net = Net().to(DEVICE)
#     trainloader = trainloaders[int(cid)]
#     valloader = valloaders[int(cid)]
#     return FlowerClient(cid, net, trainloader, valloader)

# from flwr.common import (Code, EvaluateIns, EvaluateRes,
#                          GetParametersIns, GetParametersRes,
#                          FitIns, FitRes, Status, ndarrays_to_parameters,
#                          parameters_to_ndarrays)
#
# class FlowerClient(fl.client.Client):
#     def __init__(self, cid, net, trainloader, valloader):
#         self.cid = cid
#         self.net = net
#         self.trainloader = trainloader
#         self.valloader = valloader
#
#     def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
#         print(f"[Client {self.cid}] get_parameters")
#         ndarray: List[np.ndarray] = get_parameters(self.net)
#         parameters = ndarrays_to_parameters(ndarray)
#         status = Status(code=Code.OK, message='success')
#
#         return GetParametersRes(
#             status= status,
#             parameters= parameters
#         )
#
#     def fit(self, ins: FitIns) -> FitRes:
#         print(f"[Client {self.cid}] fit, config: {ins.config}")
#
#         # Deserialize parameters to NumPy ndarray's
#         parameters_original = ins.parameters
#         ndarrays_original = parameters_to_ndarrays(parameters_original)
#
#         # Update local model, train, get updated parameters
#         set_parameters(self.net, ndarrays_original)
#         train(self.net, self.trainloader, epochs=1)
#         ndarrays_updated = get_parameters(self.net)
#
#         # Serialize ndarray's into a Parameters object
#         parameters_updated = ndarrays_to_parameters(ndarrays_updated)
#
#         # Build and return response
#         status = Status(code=Code.OK, message="Success")
#         return FitRes(
#             status=status,
#             parameters=parameters_updated,
#             num_examples=len(self.trainloader),
#             metrics={},
#         )
#
#     def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
#         print(f"[Client {self.cid}] evaluate, config: {ins.config}")
#
#         # Deserialize parameters to NumPy ndarray's
#         parameters_original = ins.parameters
#         ndarrays_original = parameters_to_ndarrays(parameters_original)
#
#         set_parameters(self.net, ndarrays_original)
#         loss, accuracy = test(self.net, self.valloader)
#
#         # Build and return response
#         status = Status(code=Code.OK, message="Success")
#         return EvaluateRes(
#             status=status,
#             loss=float(loss),
#             num_examples=len(self.valloader),
#             metrics={"accuracy": float(accuracy)},
#         )





# Create FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
#     min_fit_clients=10,  # Never sample less than 10 clients for training
#     min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
#     min_available_clients=10,  # Wait until all 10 clients are available
# )
#
# # Specify the resources each of your clients need. By default, each
# # client will be allocated 1x CPU and 0x GPUs
# client_resources = None
# if DEVICE.type == "cuda":
#     client_resources = {"num_gpus": 1}
#
#
# # Start simulation
# fl.simulation.start_simulation(
#     client_fn=numpyclient_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=strategy,
#     client_resources=client_resources,
# )