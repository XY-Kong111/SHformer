import os
import numpy as np
import torch
import torch_model
import client
import server
import flwr as fl
import random
from TimesNet import Federated_main
from torch.utils.tensorboard import SummaryWriter


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)



DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def numpyclient_lstm(cid)->client.FlowerNumPyClient:
    setting = 'LSTM'
    net=torch_model.LstmNet().to(DEVICE)
    test_dir=r'..\dataset\test\normalize'
    trainloaders,valloaders,testloaders,num_clients=torch_model.load_datasets(test_dir,window_size=96,selected_feature=['Gross Load (MW)','Heat Input (mmBtu)'],
    selected_label=['CO2 Mass (short tons)'])

    trainloader=trainloaders[int(cid)]
    valloader=valloaders[int(cid)]
    testloader=testloaders[int(cid)]
    return client.FlowerNumPyClient(cid,net,trainloader,valloader,testloader, setting).to_client()

def numpyclient_cnn(cid)->client.FlowerNumPyClient:
    setting = 'CNN'
    net=torch_model.CnnNet().to(DEVICE)
    test_dir=r'..\dataset\test\normalize'
    trainloaders,valloaders,testloaders,num_clients=torch_model.load_datasets(test_dir,window_size=24,selected_feature=['Gross Load (MW)','Heat Input (mmBtu)'],
    selected_label=['CO2 Mass (short tons)'])

    trainloader=trainloaders[int(cid)]
    valloader=valloaders[int(cid)]
    testloader=testloaders[int(cid)]
    return client.FlowerNumPyClient(cid,net,trainloader,valloader,testloader, setting).to_client()

def numpyclient_mlp(cid)->client.FlowerNumPyClient:
    setting = 'MLP'
    net=torch_model.MlpNet().to(DEVICE)
    test_dir=r'..\dataset\test\normalize'
    trainloaders,valloaders,testloaders,num_clients=torch_model.load_datasets(test_dir,window_size=24,selected_feature=['Gross Load (MW)','Heat Input (mmBtu)'],
    selected_label=['CO2 Mass (short tons)'])

    trainloader=trainloaders[int(cid)]
    valloader=valloaders[int(cid)]
    testloader=testloaders[int(cid)]
    return client.FlowerNumPyClient(cid,net,trainloader,valloader,testloader, setting).to_client()

def numpyclient_timen(cid)->client.FlowerNumPyClient:
    setting = 'TimesNet'
    # print(r'successful!')
    exp, model=Federated_main.main(tran_arg)
    trainloader=trainloaders[int(cid)]
    valloader=valloaders[int(cid)]
    testloader=testloaders[int(cid)]
    return client.FlowerNumPyClientTrans(cid,exp,model,trainloader,valloader,testloader, setting).to_client()


if __name__ == '__main__':
    dl_family = {
        'LSTM': numpyclient_lstm,
        'CNN': numpyclient_cnn,
        'MLP': numpyclient_mlp,
        'TimesNet': numpyclient_timen,
    }
    transformer_based = ['TimesNet', 'Informer', 'Transformer' ]
    model_name = 'LSTM'

    if model_name in transformer_based:
        trainloaders = []
        valloaders = []
        testloaders = []
        fold_path = r'../dataset/test/normalize//'
        fold_name = os.listdir(fold_path)[0]
        tran_arg = Federated_main.tranformer_paramemters(model_name, root_dir=fold_path, data_path=None)
        for i in os.listdir(fold_path):
            tran_arg.data_path = i
            trainset, valset, testset = Federated_main.main(tran_arg, is_built=False)
            trainloaders.append(trainset)
            valloaders.append(valset)
            testloaders.append(testset)

    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}
    num_clients = 4

    save_dir = os.path.join('.\log/CO2_emission_acconting/federated', 'LSTM')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    total_writer = SummaryWriter(save_dir)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=dl_family[model_name],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=server.strategy,
        client_resources=client_resources,
    )
    total_writer.close()