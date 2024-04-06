import numpy as np
import torch


def absolute_error(pred, label, missing_value=0):  #计算预测值和实际值的绝对误差
    if pred.device != label.device:
        label = label.to(pred.device)
    return ((pred - label) * get_valid(label, missing_value)).abs().sum()


def square_error(pred, label, missing_value=0):  #计算预测值和实际值的均方误差
    if pred.device != label.device:
        label = label.to(pred.device)
    return (((pred - label) ** 2) * get_valid(label, missing_value)).sum()


def get_valid(label, missing_value=0):   #符合规定的label才有效
    return ((label - missing_value).abs() > 1e-8).to(dtype=torch.float)


def num_valid(label, missing_value=0):  #计算符合规定的预测数目
    return get_valid(label, missing_value).sum()


# def masked_mae(pred, label, missing_value=0):  #计算绝对误差的平均值
#     if pred.device != label.device:
#         label = label.to(pred.device)
#     return absolute_error(pred, label, missing_value) / (num_valid(label, missing_value) + 1e-8)

#损失函数
class Metric:
    @staticmethod
    def create_metric(name):
        if name == 'mae': return MetricMAE()
        if name == 'rmse': return MetricRMSE()
        return None

    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.value = 0

    def update(self, pred, label):
        raise NotImplementedError("To be implemented")

    def get_value(self):
        raise NotImplementedError("To be implemented")


class MetricMAE(Metric):
    def __init__(self):
        super(MetricMAE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += absolute_error(pred, label)

    def get_value(self):
        return (self.value / (self.cnt + 1e-8)).item()


class MetricRMSE(Metric):
    def __init__(self):
        super(MetricRMSE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += square_error(pred, label)

    def get_value(self):
        return torch.sqrt(self.value / (self.cnt + 1e-8)).item()


class Metrics:
    def __init__(self, metric_list, metric_index):
        self.metric_all = {m: Metric.create_metric(m) for m in metric_list}
        self.metric_horizon = {'%s-horizon' % m: Metric.create_metric(m)  for m in metric_list}
        self.metric_index = metric_index

    def reset(self):
        for m in self.metric_all.values(): m.reset()
        for k, arr in self.metric_horizon.items():
            for m in arr:
                m.reset()

    def update(self, pred, label):
        for m in self.metric_all.values(): m.update(pred, label)
        for k, arr in self.metric_horizon.items():
            arr.update(pred, label)

    def get_value(self):
        ret = {k: np.array([m.get_value()]) for k, m in self.metric_all.items()}
        for k, arr in self.metric_horizon.items():
            ret[k] = np.array(arr.get_value())
        return ret

    def __repr__(self):
        out_str = []
        for k, v in sorted(self.get_value().items()):
            out_str += ['%s: %s' % (k, v)]
        return '\t'.join(out_str)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    if preds.device != labels.device:
        labels = labels.to(preds.device)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    if preds.device != labels.device:
        labels = labels.to(preds.device)
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    if preds.device != labels.device:
        labels = labels.to(preds.device)
    if mask.device != preds.device:
        mask = mask.to(preds.device)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    if preds.device != labels.device:
        labels = labels.to(preds.device)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
