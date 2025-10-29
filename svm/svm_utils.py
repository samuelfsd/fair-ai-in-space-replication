# svm_utils.py

import numpy as np
import tensorflow as tf

def get_prf(true_class, total_class, pred_class):
    pre = true_class / pred_class
    rec = true_class / total_class

    pre_fix = np.nan_to_num(pre, nan=0.0)
    rec_fix = np.nan_to_num(rec, nan=0.0)

    epsilon = 1e-10
    f1 = 2 / (pre_fix ** (-1) + rec_fix ** (-1) + epsilon)

    f1 = np.nan_to_num(f1, nan=0.0)

    return pre, rec, f1

def get_avg_f1(f1, total_class):
    avg_f1 = np.sum(f1 * total_class / np.sum(total_class))
    return avg_f1

def get_fairness_loss_all_sklearn(y_true_one_hot, y_pred_labels, partitionings_list, all_partitioning_data_list):
    """
    Calcula a perda de fairness (desvio padrão dos F1-scores)
    iterando por todas as partições e seus subgrupos.
    """

    true_pred_w_class, pred_w_class_one_hot = get_class_wise_list_sklearn(y_true_one_hot, y_pred_labels)

    all_w_f1_scores = []

    for (index1, index2) in partitionings_list:

        data_list = all_partitioning_data_list[(index1, index2)]
        num_subgroups = index1 * index2

        for i in range(num_subgroups):
            indices_particao = data_list[i]

            if len(indices_particao) == 0:
                all_w_f1_scores.append(0.0)
                continue

            true_class_part = np.sum(true_pred_w_class[indices_particao], axis=0).reshape(-1)
            total_class_part = np.sum(y_true_one_hot[indices_particao], axis=0).reshape(-1)
            total_pred_part = np.sum(pred_w_class_one_hot[indices_particao], axis=0).reshape(-1)

            pre, rec, f1 = get_prf(true_class_part, total_class_part, total_pred_part)

            w_f1 = get_avg_f1(f1, total_class_part)
            all_w_f1_scores.append(w_f1)

    return np.std(all_w_f1_scores)

def get_class_wise_accuracy_sklearn(y_true_one_hot, y_pred_labels, prf=False):
    """
    Calcula acurácia para sklearn, onde y_true é one-hot mas y_pred são labels 1D.
    """
    num_class = y_true_one_hot.shape[1]
    y_true_labels = np.argmax(y_true_one_hot, axis=1)

    stat = (y_true_labels == y_pred_labels).astype(float) # 1.0 se correto, 0.0 se incorreto

    true_pred_w_class = y_true_one_hot * np.expand_dims(stat, 1)
    true = np.sum(true_pred_w_class, axis=0).reshape(-1)
    total = np.sum(y_true_one_hot, axis=0).reshape(-1)

    if prf:
        pred_w_class = tf.one_hot(y_pred_labels, depth=num_class).numpy()
        pred_total = np.sum(pred_w_class, axis=0).reshape(-1)
        return true, total, pred_total
    else:
        return true, total

def get_class_wise_list_sklearn(y_true_one_hot, y_pred_labels):
    """
    Lista de acertos/erros para sklearn.
    """
    num_class = y_true_one_hot.shape[1]
    y_true_labels = np.argmax(y_true_one_hot, axis=1)

    y_pred_one_hot = tf.one_hot(y_pred_labels, depth=num_class).numpy()

    stat = (y_true_labels == y_pred_labels).astype(float)
    true_pred_w_class = y_true_one_hot * np.expand_dims(stat, 1)

    return true_pred_w_class, y_pred_one_hot

def get_weighted_f1_one_sklearn(model, X_data, y_data_one_hot, all_partitioning_data_list, index1, index2):
    """
    Calcula o F1 ponderado para partições específicas usando um modelo sklearn.
    """
    w_f1_list = np.zeros((index1 * index2), dtype='float')

    y_pred_labels = model.predict(X_data)

    true_pred_w_class, pred_w_class_one_hot = get_class_wise_list_sklearn(y_data_one_hot, y_pred_labels)

    data_list = all_partitioning_data_list[(index1, index2)]

    for i in range(index1 * index2):
        indices_particao = data_list[i]
        true_class_part = np.sum(true_pred_w_class[indices_particao], axis=0).reshape(-1)
        total_class_part = np.sum(y_data_one_hot[indices_particao], axis=0).reshape(-1)
        total_pred_part = np.sum(pred_w_class_one_hot[indices_particao], axis=0).reshape(-1)

        pre, rec, f1 = get_prf(true_class_part, total_class_part, total_pred_part)
        w_f1_list[i] = get_avg_f1(f1, total_class_part)

    return w_f1_list