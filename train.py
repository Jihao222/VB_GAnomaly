import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 20

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    edge_att_weights_all = []

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()
            # attack_labels为0
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
            # x=(batch，节点数量，一个slide_win长度的数据)  labels=这组数据需要预测的数据(batch,node_num)  edge_index=边(batch,node_num,所有边的数量)

            optimizer.zero_grad()
            out1, edge_att_weight, new_edge_index0 = model(x, edge_index)
            out = out1.float().to(device)  # (1,4)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1
            edge_att_weights_all = edge_att_weights_all + edge_att_weight  # list:154

        edge_att_weights = new_edge_index0 + edge_att_weights_all
        Fi_edge_att_weights = pd.DataFrame(data=edge_att_weights)
        now1 = datetime.now()
        savestr = now1.strftime('%m.%d-%H_%M_%S')
        Fi_edge_att_weights.to_csv(f'data\{dataset_name}\Att_results-train\Att_weights_{savestr}.csv')

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, weights_edge_index, att_weights, val_result = test(model, val_dataloader)  # 2个

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss



    return train_loss_list
