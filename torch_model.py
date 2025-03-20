from Evaluation_metrics import EvaluationMetrics
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU


def normalize(data):
    data_np = np.array(data)
    return (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np))


def row_data_preprocess(dir: str, window_size: int, selected_features: List[str],
                        selected_label: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(dir)

    # Step 1: Drop the 'Unnamed: 0' column
    # df_data= df_data.drop(columns=['Unnamed: 0'])
    # Step 2: Ensure the data types are correct, especially the Date column
    df['Date'] = pd.to_datetime(df['Date'])
    # Before cleaning, sort the data by 'Date' and 'Hour'
    df_data = df.sort_values(by=['Date', 'Hour'])
    # print(df_data)
    # Step 3: Optimize data types
    # For this step, we'll convert columns that are clearly integers but may be stored as floats
    integer_columns = ['Gross Load (MW)', 'Steam Load (1000 lb/hr)']
    df_data[integer_columns] = df_data[integer_columns].apply(pd.to_numeric, downcast='integer')
    df_data[integer_columns[0]] = df_data[integer_columns[0]] + df_data[integer_columns[1]]
    df_final = df_data[selected_features]

    # Convert the dataframe to a numpy array
    # print(df_final)
    np_final = df_final.to_numpy()

    shape = (np_final.shape[0] - window_size + 1, window_size, np_final.shape[-1])
    strides = (np_final.strides[0], np_final.strides[0], np_final.strides[1])
    np_input = np.lib.stride_tricks.as_strided(np_final, shape=shape, strides=strides, writeable=False)
    np_input = np.delete(np_input, obj=-1, axis=0)

    np_output = np.delete(df_data[selected_label].to_numpy(), slice(0, window_size), axis=0)

    # print(np_input.shape, '\n', np_output.shape)
    return np_input, np_output


def load_datasets(dataset_dir, window_size, selected_feature, selected_label) \
        -> Tuple[list[DataLoader], list[DataLoader], list[DataLoader], int]:
    trainloaders = []
    valloaders = []
    testloaders = []
    x = []
    y = []
    length = []

    for i in os.listdir(dataset_dir):
        path = dataset_dir + '\\' + i
        im_x, im_y = row_data_preprocess(path, window_size, selected_feature, selected_label)
        x.append(im_x);
        y.append(im_y);
        length.append(im_x.shape[0])

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    # print(f"Concatenated input{x}, and concatenated output {y}")
    x[:, :, 0] = normalize(x[:, :, 0])
    x[:, :, 1] = normalize(x[:, :, 1])
    y = normalize(y)
    # print(x,'\n',y)

    # 转换为PyTorch张量
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    # print(f"normalized input{x}, and normalized output {y}")
    dataset = TensorDataset(x, y)

    indexes = [sum(length[:i]) for i in range(len(length) + 1)]
    subsets = [Subset(dataset, range(indexes[i], indexes[i + 1])) for i in range(len(length))]

    # print(f"splited subsets {subsets[0][0:1]} and {subsets[1][-1]}")

    for ds in subsets:
        len_test = len(ds) // 10  # 10 % testing set
        len_train = len(ds) - len_test
        # print(len_train)
        testset = Subset(ds, range(len_train, len(ds)))
        trainset = Subset(ds, range(0, len_train))
        # print(f"splited testset {testset[0][0:1]} and {testset[1][-1]}")
        lengths = [len_train - len_test, len_test]

        ds_train, ds_val = random_split(trainset, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
        testloaders.append(DataLoader(testset, batch_size=32))
    return trainloaders, valloaders, testloaders, len(os.listdir(dataset_dir))


class LstmNet(nn.Module):
    def __init__(self) -> None:
        super(LstmNet, self).__init__()
        self.lstm1 = nn.LSTM(2, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 512, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)

        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)

        # 只取LSTM输出的最后一步
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class CnnNet(nn.Module):
    def __init__(self) -> None:
        super(CnnNet, self).__init__()
        self.model1 = nn.Sequential(
            # 输入通道一定为1，输出通道为卷积核的个数，2为卷积核的大小（实际为一个[1,2]大小的卷积核）
            nn.Conv1d(2, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32,3),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 输出大小：torch.Size([128, 32])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=640, out_features=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        cnn_out = self.model1(x)
        # print(cnn_out.size())
        predictions =  self.model2(cnn_out)
        return predictions

class MlpNet(nn.Module):
    def __init__(self) -> None:
        super(MlpNet, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(48, 128,  bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        predictions = self.model1(x)
        # print(cnn_out.size())
        return predictions

# def train(net, trainloader, valloader,epochs: int ):
#     """Train the network on the training set."""
#     criterion = nn.MSELoss()  # 假设我们的任务是回归问题
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#
#
#     writer = SummaryWriter('log/CO2_emission_acconting')
#
#
#     net.train()
#     for epoch in range(epochs):
#         loss, train_loss, total = 0, 0, 0
#         for time_series, true_v in trainloader:
#             time_series, true_v = time_series.to(DEVICE), true_v.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(net(time_series), true_v)
#             loss.backward()
#             optimizer.step()
#         # Metrics
#         train_loss += loss
#          # 验证阶段
#         net.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for time_series, true_v in valloader:
#                 outputs = net(time_series)
#                 val_loss += criterion(outputs, true_v).item()
#
#         train_loss /= len(trainloader.dataset)
#         val_loss /= len(valloader.dataset)
#         print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
#
#         # 写入到TensorBoard
#         writer.add_scalar('training loss', loss, epoch)
#         writer.add_scalar('validation loss', val_loss, epoch)
#
#     # 关闭TensorBoard写入器
#     writer.close()
#
#     return  train_loss, val_loss

def train(net, trainloader, valloader, epochs: int, writer,learning_rate):
    """Train the network on the training set."""
    criterion = nn.MSELoss()  # 假设我们的任务是回归问题
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()
    for epoch in range(epochs):
        loss, train_loss, total = 0, 0, 0
        for time_series, true_v in trainloader:
            time_series, true_v = time_series.to(DEVICE), true_v.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(time_series), true_v)
            loss.backward()
            optimizer.step()
        # Metrics
        train_loss += loss
        # 验证阶段
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for time_series, true_v in valloader:
                outputs = net(time_series)
                val_loss += criterion(outputs, true_v).item()

        train_loss /= len(trainloader.dataset)
        val_loss /= len(valloader.dataset)
        print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        # 写入到TensorBoard
        writer.add_scalar('training loss', loss)
        writer.add_scalar('validation loss', val_loss)
    # 关闭TensorBoard写入器
    writer.close()

    return train_loss, val_loss


def test(net, testloader, writer, dir, epoch):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    true = []
    monitor = []
    with torch.no_grad():
        for time_series, true_v in testloader:
            time_series, true_v = time_series.to(DEVICE), true_v.to(DEVICE)
            outputs = net(time_series)
            loss += criterion(outputs, true_v).item()
            true.extend(true_v.detach().numpy())
            monitor.extend(outputs.detach().numpy())
    f_true = np.array(true).squeeze().transpose()*2011.8
    f_true = np.where(f_true <= 0, 0.0001, f_true)
    # print(f_true.shape)
    f_monitor = np.array(monitor).squeeze().transpose()*2011.8
    f_monitor = np.where(f_monitor <= 0, 0.0001, f_monitor)

    Evaluation = EvaluationMetrics()
    RMSE = Evaluation.rmse(f_true, f_monitor)
    MAE = Evaluation.mae(f_true, f_monitor)
    MAPE = Evaluation.mape(f_true, f_monitor)
    nRMSE = Evaluation.nrmse(f_true, f_monitor)
    R2 = Evaluation.R2(f_true, f_monitor)
    length =len(f_true)-1
    data = {
        'ground_truth': f_true,
        'monitor_result': f_monitor,
        'rmse': [RMSE]+[np.nan]*length,
        'mae': [MAE]+[np.nan]*length,
        'mape': [MAPE]+[np.nan]*length,
        'nrmse': [nRMSE]+[np.nan]*length,
        'r2': [R2]+[np.nan]*length
    }

    pd.DataFrame(data).to_excel(f'{dir}/{epoch}results.xlsx',index=False)
    # loss /= len(testloader.dataset)
    # writer.add_scalar('test loss', loss)
    print(f' Test Loss: {loss}')

    return loss


if __name__ == '__main__':
    # test_dir = r'..\dataset\carbon_monitoring_dataset\Fort Martin Power Station.xlsx'
    # row_data_preprocess(test_dir, window_size=24, selected_features=['Gross Load (MW)'
    #     , 'Heat Input (mmBtu)'],selected_label=['CO2 Mass (short tons)'])
    # test_dir = r'..\dataset\carbon_monitoring_dataset'
    test_dir = r'../dataset/test/normalize/'
    save_dir= r'.\log\CO2_emission_acconting\localized\LSTM'
    trainset, valset, testset, num_clients = load_datasets(test_dir, window_size=96, selected_feature=['Gross Load (MW)'
        , 'Heat Input (mmBtu)'], selected_label=['CO2 Mass (short tons)'])
    # net = LstmNet()

    net = LstmNet()
    # client_number = ['a', 'b']
    # for batch_idx, (data, target) in enumerate(testset):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Data: {data}")
    #     print(f"  Target: {target}")
    train_set = ConcatDataset(trainset)
    train_set = DataLoader(trainset, batch_size=32, shuffle=True)
    val_set = ConcatDataset(valset)
    val_set = DataLoader(val_set, batch_size=32, shuffle=True)
    test_set = ConcatDataset(testset)
    test_set = DataLoader(test_set, batch_size=32, shuffle=True)
    # print(train_set.size())

    writer_lol = SummaryWriter(save_dir)
    train(net, train_set, val_set, 10, writer_lol)
    test(net, test_set, writer_lol, save_dir)
    writer_lol.close()

