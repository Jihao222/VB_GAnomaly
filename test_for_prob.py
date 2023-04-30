import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from util.data import *
from util.preprocess import *

import csv

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    edge_att_weights_all = []

    test_len = len(dataloader)

    model.eval()  # best_model
    enable_dropout(model)

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        # x:(128,27,15)  y:(128,27)即实际需要预测的数值  labels：(128)即实际的标签
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
            predicted_0, edge_att_weight, new_edge_index0 = model(x, edge_index)
            predicted = predicted_0.float().to(device)

            loss = loss_func(predicted, y)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        edge_att_weights_all = edge_att_weights_all + edge_att_weight

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # edge_att_weights = new_edge_index0 + edge_att_weights_all
    # Fi_edge_att_weights = pd.DataFrame(data=edge_att_weights)
    # now1 = datetime.now()
    # savestr = now1.strftime('%m.%d-%H_%M_%S')
    # Fi_edge_att_weights.to_csv('data\P_0.1mpa-2\Att_results-test\Att_weights_{}.csv'.format(savestr))

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    # now = datetime.now()
    # savestr = now.strftime('%m.%d-%H_%M_%S')
    # test_predicted = pd.DataFrame(test_predicted_list)
    # test_predicted.to_csv('data\P_0.1mpa-2\Results2\Test_predicted_{}.csv'.format(savestr), index=False)
    # test_ground = pd.DataFrame(test_ground_list)
    # test_ground.to_csv('data\P_0.1mpa-2\Results2\Test_ground_list_{}.csv'.format(savestr), index=False)
    # test_labels = pd.DataFrame(test_labels_list)
    # test_labels.to_csv('data\P_0.1mpa-2\Results2\Test_labels_list_{}.csv'.format(savestr), index=False)

    return avg_loss, new_edge_index0, edge_att_weights_all, [test_predicted_list, test_ground_list, test_labels_list]