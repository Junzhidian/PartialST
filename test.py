import torch
from torch.utils.data import DataLoader
from PartialST.models.partialst import PartialST
from data.dataset import DatasetFactory
import numpy as np
import math
import time

seed = 777
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DataConfiguration:
    # Data
    name = 'TaxiBJ'
    portion = 1.  # portion of data

    len_close = 3
    len_period = 1
    len_trend = 1
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = 1
    interval_trend = 7

    ext_flag = True
    timeenc_flag = 'w'  # 'm', 'w', 'd'
    rm_incomplete_flag = True
    fourty_eight = True
    previous_meteorol = True

    ext_dim = 77 # 77
    dim_flow = 2
    dim_h = 32
    dim_w = 32

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test():
    set_seed(seed)
    dconf = DataConfiguration()
    ds_factory = DatasetFactory(dconf)
    select_pre = 0
    test_ds = ds_factory.get_test_dataset(select_pre)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=32,
        shuffle=False
    )

    model = PartialST(dconf)
    print('num parameters:',count_parameters(model))
    model.load("./reports/TaxiBJ_3_1_1_12/partialst/model.best.pth")
    model = model.to(device)

    model.eval()
    trues = []
    preds = []
    with torch.no_grad():
        for _, (X, X_ext, Y, Y_ext) in enumerate(test_loader):
            X = X.to(device)  
            X_ext = X_ext.to(device)  
            Y = Y.to(device)  
            Y_ext = Y_ext.to(device) 
            outputs = model(X, X_ext, Y_ext)
            true = ds_factory.ds.mmn.inverse_transform(Y.detach().cpu().numpy())
            pred = ds_factory.ds.mmn.inverse_transform(outputs.detach().cpu().numpy())
            trues.append(true)
            preds.append(pred)
    trues = np.concatenate(trues, 0)
    preds = np.concatenate(preds, 0)
    mae = np.mean(np.abs(preds-trues))
    rmse = np.sqrt(np.mean((preds-trues)**2))
    print("test_rmse: %.4f, test_mae: %.4f" % (rmse, mae))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def elapsed_time_format(total_time):
    hour = 60 * 60
    minute = 60
    if total_time < 60:
        return f"{math.ceil(total_time):d} secs"
    elif total_time > hour:
        hours = divmod(total_time, hour)
        return f"{int(hours[0]):d} hours, {elapsed_time_format(hours[1])}"
    else:
        minutes = divmod(total_time, minute)
        return f"{int(minutes[0]):d} mins, {elapsed_time_format(minutes[1])}"

if __name__ == '__main__':
    print("l_c:", DataConfiguration.len_close, "l_p:", DataConfiguration.len_period, "l_t:", DataConfiguration.len_trend)
    
    start_time = time.time()
    test()
    print(f"Elapsed time: {elapsed_time_format(time.time() - start_time)}")
    