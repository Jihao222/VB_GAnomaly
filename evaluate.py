from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import pandas as pd
from datetime import datetime

def get_err_mean_and_var(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))  # np.abs:取绝对值

    err_mean = np.mean(np_arr)
    err_var = np.var(np_arr)

    return err_mean, err_var

def get_proportional_err(predicted):
    test_predict, test_gt = predicted
    np_arr = np.abs(np.subtract(np.array(test_predict), np.array(test_gt)))  # np.abs:取绝对值
    epsilon = 1e-3
    err_pro = np_arr/(test_gt+epsilon)

    return err_pro

def get_full_err_scores(test_result, val_result):  # 异常评分:将t时刻的预期行为与观测到的行为(实际行为)进行比较，计算出t时刻的传感器i的错误值Err
    # test_result：[test_predicted_list, test_ground_list, test_labels_list]
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]  # 测试集结果中传入的第i个传感器的预测值和实际值
        val_re_list = np_val_result[:2,:,i]  # 验证集中传入的第i个传感器的预测值和实际值

        scores = get_err_scores(test_re_list, val_re_list)  # 根据测试结果得到的异常分数
        scores_prop = get_proportional_err(test_re_list)  # 根据测试结果得到的异常比率分数
        normal_dist = get_err_scores(val_re_list, val_re_list)  # 根据验证集的结果得到的异常分数
        normal_prop= get_proportional_err(val_re_list)  # 根据验证集的结果得到的异常比率分数

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
            all_scores_prop = scores_prop
            all_normals_prop = normal_prop
        else:
            all_scores = np.vstack((  # 在竖直方向上拼接数组all_scores与scores
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))
            all_scores_prop = np.vstack((  # 在竖直方向上拼接数组all_scores与scores
                all_scores_prop,
                scores_prop
            ))
            all_normals_prop= np.vstack((  # 在竖直方向上拼接数组all_scores与scores
                all_normals_prop,
                normal_prop
            ))

    return all_scores, all_normals, all_scores_prop, all_normals_prop  # 根据所有传感器的测试集结果、验证集结果


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals, all_scores_prop, all_normals_prop = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores  # 所有传感器的测试集结果SMA的最大值



def get_err_scores_As(test_res, val_res):  # 传入某个传感器的测试集和验证集结果中的预测值与真实值
    test_predict, test_gt = test_res  # 测试集结果的预测值、真实值
    val_predict, val_gt = val_res

    #n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)  # 测试集中所有预测值与真实值的中位数、四分位距
    n_err_mid, n_err_iqr = get_err_mean_and_var(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))  # 公式(11)计算出的传感器i的错误值Err
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) +epsilon)  # (12)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])  # 使用简单移动平均生成平滑的分数As。对测试结果的错误值Err的每四个数求平均

    return smoothed_err_scores

def get_err_scores(test_res, val_res):  # 传入某个传感器的测试集和验证集结果中的预测值与真实值
    test_predict, test_gt = test_res  # 测试集结果的预测值、真实值
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)  # 测试集中所有预测值与真实值的中位数、四分位距
    #n_err_mid, n_err_iqr = get_err_mean_and_var(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64),
                        np.array(test_gt).astype(np.float64)
                    ))  # 公式(11)计算出的传感器i的错误值Err
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) +epsilon)  # (12)

    return err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return [f1, pre, rec, auc_score, thresold], topk_indices, pred_labels, total_topk_err_scores


def get_best_performance_data(total_err_scores, gt_labels, topk=1):  # (全部传感器的测试集结果的错误值的平滑的分数As，测试集的真实attack标签)

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    # topk_indices：0维度上最大值所对应的索引，即异常分数最大的传感器(定位出的最大故障的节点的索引)
    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    # 为四个传感器中错误值分数的最大值
    # take_along_axis()用于由索引数组生成新的矩阵。即取出total_err_scores中0维度上topk_indices指示的值，建立的新数组

    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
    # eval_scores:试400个F1值，选一个效果最好的。final_topk_fmeas为F1值

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))  # 找到F1值最大值的索引
    thresold = thresolds[th_i]  # 分类阈值（thresold）：使得F1最大时的阈值

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1  # 预测的attack标签

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    label = list(zip(gt_labels, pred_labels))
    # labels = list(zip(*label))
    # labels = list(map(list,zip(*label)))
    #columns_name = ['gt_labels', 'pred_labels']
    #results_list = pd.DataFrame(columns=columns_name, index=None, data=label)
    #now = datetime.now()
    #datestr = now.strftime('%m.%d-%H_%M_%S')
    #dataset = self.env_config['dataset']
    #results_list.to_csv(f'.\data\{dataset}\Results\labels_{datestr}.csv')

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    # sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
    # sklearn.metrics.roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None, max_fpr=None)

    return [max(final_topk_fmeas), pre, rec, auc_score, thresold], topk_indices, pred_labels, total_topk_err_scores

