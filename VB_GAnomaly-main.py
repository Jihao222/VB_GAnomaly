# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import operator

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc

from datasets.TimeDataset import TimeDataset

from models.GAno_Ad import GAno

from train import train
from test_for_prob import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

from datetime import datetime

import os
import argparse
from pathlib import Path

import random

# 定义模型
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
        train, test= train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train',
                                    config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test',
                                    config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                           shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GAno(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']
                         ).to(self.device)

    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            results_save_path = self.get_save_path()[1]

        # test
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, val_edge_weights_index, val_att_weights, self.val_result = test(best_model, self.val_dataloader)
        _, test_result_mean = self.prob_test(best_model, self.test_dataloader)

        topk_indices, pred_labels, info = self.get_score(test_result_mean, self.val_result)

        print('结束')

    def prob_test(self, best_model, test_dataloader):

        Random_group = 1
        MCSample = 300
        PredTime = 870
        dataset = self.env_config['dataset']
        F1_score_list = []
        precision_list = []
        recall_list = []
        auc_score_list = []
        F1_score_list2 = []
        precision_list2 = []
        recall_list2 = []
        auc_score_list2 = []

        for random_sample in range(Random_group):
            test_predicted_s1 = np.empty((0, PredTime))
            test_predicted_s2 = np.empty((0, PredTime))
            test_predicted_s3 = np.empty((0, PredTime))
            test_predicted_s4 = np.empty((0, PredTime))
            pred_labels_AllSample = np.empty((0, PredTime))
            sensor_indices = np.empty((0, PredTime))
            max_err_scores = np.empty((0, PredTime))
            max_err_scores2 = np.empty((0, PredTime))
            edge_weights = np.empty((0, 870, 12))
            for MC_sample in range(MCSample):
                now = datetime.now()
                datestr1 = now.strftime('%m.%d-%H.%M.%S.%f')
                if MC_sample == 0:
                    time_in = datestr1
                elif MC_sample == MCSample-1:
                    time_out = datestr1
                _, edge_weights_index, att_weights, test_result = test(best_model, test_dataloader)

                edge_att_weights = np.array(att_weights)
                edge_weights = np.vstack((edge_weights, edge_att_weights[np.newaxis, :, :]))

                topk_indices, total_topk_err_scores, pred_labels, info, topk_indices2, total_topk_err_scores2, \
                pred_labels2, info2 = self.get_score(test_result, self.val_result, MC_sample)
                F1_score_list.append(info[0])
                precision_list.append(info[1])
                recall_list.append(info[2])
                auc_score_list.append(info[3])

                F1_score_list2.append(info2[0])
                precision_list2.append(info2[1])
                recall_list2.append(info2[2])
                auc_score_list2.append(info2[3])

                pred_labels_AllSample = np.vstack((pred_labels_AllSample, pred_labels))
                sensor_indices = np.vstack((sensor_indices, topk_indices))
                max_err_scores = np.vstack((max_err_scores, total_topk_err_scores))
                max_err_scores2 = np.vstack((max_err_scores2, total_topk_err_scores2))

                np_test_result = np.array(test_result)
                test_predicted_s1 = np.vstack((test_predicted_s1, np_test_result[0, :, 0]))
                test_predicted_s2 = np.vstack((test_predicted_s2, np_test_result[0, :, 1]))
                test_predicted_s3 = np.vstack((test_predicted_s3, np_test_result[0, :, 2]))
                test_predicted_s4 = np.vstack((test_predicted_s4, np_test_result[0, :, 3]))

                F1_score_list.append(info[0])
                precision_list.append(info[1])
                recall_list.append(info[2])
                auc_score_list.append(info[3])

        auc_score = np.array(auc_score_list)
        auc_max_index, auc_max_number = max(enumerate(auc_score), key=operator.itemgetter(1))
        auc_min_index, auc_min_number = min(enumerate(auc_score), key=operator.itemgetter(1))
        auc_mean = np.mean(auc_score)
        print(f'auc_max_index={auc_max_index},auc_max_number={auc_max_number}')
        print(f'auc_min_index={auc_min_index},auc_min_number={auc_min_number}')
        print(f'auc_mean={auc_mean}')

        result_mean_s1 = np.mean(test_predicted_s1, axis=0)
        result_mean_s2 = np.mean(test_predicted_s2, axis=0)
        result_mean_s3 = np.mean(test_predicted_s3, axis=0)
        result_mean_s4 = np.mean(test_predicted_s4, axis=0)
        predicted_mea = np.vstack((result_mean_s1, result_mean_s2, result_mean_s3, result_mean_s4))
        predicted_mea = predicted_mea.transpose()
        predicted_mean = predicted_mea.tolist()

        test_observed = np_test_result[1, :, :]
        test_labels = np_test_result[2, :, :]
        test_result_mean = [predicted_mean, test_observed.tolist(), test_labels.tolist()]  #

        result_std_s1 = np.std(test_predicted_s1, axis=0)
        result_std_s2 = np.std(test_predicted_s2, axis=0)
        result_std_s3 = np.std(test_predicted_s3, axis=0)
        result_std_s4 = np.std(test_predicted_s4, axis=0)
        predicted_std = (np.vstack((result_std_s1, result_std_s2, result_std_s3, result_std_s4))).transpose()

        return _, test_result_mean

    def get_position(self, topk_indices, pred_labels, att_weights):
        dataset = self.env_config['dataset']
        att_weights = np.array(att_weights)
        feature_num = len(self.feature_map)

        sensor = []
        index_temp = []
        Sensor_Predictions = []
        Att_weights_Predictions = np.empty((0, 12))
        for index, label in enumerate(pred_labels):
            if label == 1:
                sensor.append(topk_indices[0, index])
                Sensor_Predictions.append(topk_indices[0, index])
                Att_weights_Predictions = np.vstack((Att_weights_Predictions, att_weights[index, :]))
                index_temp.append(index)
        Sensor_Prediction = np.argmax(np.bincount(np.array(Sensor_Predictions)))
        Att_weights_Prediction = np.mean(Att_weights_Predictions, axis=0)


        return Sensor_Prediction, Att_weights_Prediction, index_temp

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[
                          val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                      shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                    shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result, MC_sample):
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores, test_scores_prop, test_normals_prop = get_full_err_scores(test_result, val_result)

        top1_best_info, topk_indices, pred_labels, total_topk_err_scores = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_best_info2, topk_indices2, pred_labels2, total_topk_err_scores2 = get_best_performance_data(test_scores_prop,
                                                                                                     test_labels,
                                                                                                     topk=1)

        top1_val_info, topk_indices, pred_labels, total_topk_err_scores = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)
        top1_val_info2, topk_indices2, pred_labels2, total_topk_err_scores2 = get_val_performance_data(test_scores_prop,
                                                                                                   test_normals_prop,
                                                                                                   test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
            info2 = top1_best_info2
        elif self.env_config['report'] == 'val':
            info = top1_val_info
            info2 = top1_val_info2

        print(f'MC_sample: {MC_sample}')
        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        print(f'auc_score: {info[3]}')
        print(f'thresold: {info[4]}')

        print(f'F1 score: {info2[0]}')
        print(f'precision: {info2[1]}')
        print(f'recall: {info2[2]}\n')
        print(f'auc_score: {info2[3]}')
        print(f'thresold: {info2[4]}')

        return topk_indices, total_topk_err_scores, pred_labels, info, topk_indices2, total_topk_err_scores2, pred_labels2, info2

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m.%d-%H_%M_%S')
        datestr = self.datestr

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-epoch', help='train epoch', type=int, default=20)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=10)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=10)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='P_0.1')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='P_0.1mpa-4')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=1)
    parser.add_argument('-comment', help='experiment comment', type=str, default='P_0.1mpa-4')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=201)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('-topk', help='topk num', type=int, default=3)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='./pretrained/pre.pt')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()