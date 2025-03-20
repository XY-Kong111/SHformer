import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

class EvaluationMetrics:
    def __init__(self):
        # You can add any necessary initializations here
        pass

    def mse(self,pred, true):
        return np.mean((pred - true) ** 2)

    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))

    def calculate_rmse(self, predictions, targets):

        if len(predictions) != len(targets):
            raise ValueError("The shape of the predicted and true values do not match.")

        # 将数组转换为 NumPy 数组
        predictions = np.array(predictions)
        targets = np.array(targets)

        # 计算预测值与真实值之间的差值
        errors = predictions - targets

        # 计算均方根误差
        rmse = np.sqrt(np.mean(errors ** 2))

        return rmse

    def nrmse(self, y_true, y_pred):
        rmse2 = self.rmse(y_true, y_pred)
        return rmse2 / np.mean(y_true, axis=0)

    def mape(self, actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual))

    def smape(self, y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    def R2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def mae(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def _error(self, actual, predicted):
        """ Simple error """
        return actual - predicted



    def _percentage_error(self, actual, predicted):
        """ Percentage error """
        return (actual - predicted) / actual