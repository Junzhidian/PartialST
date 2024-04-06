import torch
import time
from torch.utils.data import DataLoader
from models.partialst import PartialST
from data.dataset import DatasetFactory
import numpy as np
import torch.nn as nn
import os
import math
import logging
import torch.optim.lr_scheduler
from utils import EarlyStopping


seed = 777

torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda")
else:
    device = 'cpu'
print(device)
epoch_nums =500
learning_rate = 0.0002
batch_size = 32
nb_residual_unit = 12
early_stop_patience = 50

m_factor = math.sqrt(1. * 16 * 8 / 81)

class DataConfiguration:
    # Data
    name = 'TaxiBJ'
    #name = 'BikeNYC'
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

    ext_dim = 77  # 77
    #ext_dim = 33
    dim_flow = 2
    dim_h = 32
    dim_w = 32
    # dim_h = 16
    # dim_w = 8



def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    logger = logging.getLogger(__name__)
    logger.info('training...')
    set_seed(seed)
    dconf = DataConfiguration()
    ds_factory = DatasetFactory(dconf)
    select_pre = 0
    train_ds = ds_factory.get_train_dataset(select_pre)
    test_ds = ds_factory.get_test_dataset(select_pre)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    train_indices = list(range(len(train_loader)))
    total_iters = np.ceil(len(train_indices) / batch_size) * epoch_nums


    out_dir = './reports/{}_{}_{}_{}_{}'.format(dconf.name,
                                                dconf.len_close,
                                                dconf.len_period,
                                                dconf.len_trend,
                                                nb_residual_unit)
    model_name = 'partialst'
    out_dir = out_dir + '/%s' % (model_name)
    os.makedirs(out_dir, exist_ok=True)


    model = PartialST(dconf)
    print('num parameters:',count_parameters(model))
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,70,80,90],gamma=0.6)
    model.to(device)
    loss_fn.to(device)
    es = EarlyStopping(patience=early_stop_patience,
                       mode='min', model=model, save_path=out_dir + '/model.best.pth')



    #b = 0.01

    rmses = [np.inf]
    for e in range(epoch_nums):
        model.train()
        for _, (X, X_ext, Y, Y_ext) in enumerate(train_loader):
            X = X.to(device)  
            X_ext = X_ext.to(device)  
            Y = Y.to(device)  
            Y_ext = Y_ext.to(device) 
            outputs = model(X, X_ext, Y_ext)
            loss1 = loss_fn(outputs, Y)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if e % 10 == 0:
            print('learning rate:', optimizer.param_groups[0]['lr'])
        its = np.ceil(len(train_indices) / batch_size) * (e + 1)  # iterations at specific epochs
        print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'
              .format(e + 1, epoch_nums, its, total_iters, loss.item()))
        model.eval()
        with torch.no_grad():
            trues = []
            preds = []
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
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        if rmse < np.min(rmses):
            f = open('{}/results.txt'.format(out_dir), 'a')
            f.write("epoch\t{}\tRMSE\t{:.6f}\tMAE\t{:.6f}\n".format(e,rmse,mae,))
            f.close()

        if es.step(rmse):
            print('early stopped! With val loss:', rmse)
            break  # early stop criterion is met, we can stop now

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
    print("l_c:", DataConfiguration.len_close, "l_p:", DataConfiguration.len_period, "l_t:",
          DataConfiguration.len_trend)
    start_time = time.time()
    train()
    print(f"Elapsed time: {elapsed_time_format(time.time() - start_time)}")
