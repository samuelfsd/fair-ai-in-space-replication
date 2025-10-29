# evaluate_svm.py

import numpy as np
import pickle
import joblib
from keras import backend as K
from svm_utils import get_class_wise_accuracy_sklearn, get_prf, get_avg_f1, get_fairness_loss_all_sklearn

# --- Globals ---
MAX_ROW_PARTITION = 5
MAX_COL_PARTITION = 5
GLOBAL_MEAN = 0.0

print("Carregando dados de teste...")
X = np.load('X_test.npy')
y_one_hot = np.load('y_test.npy')
y_labels = np.argmax(y_one_hot, axis=1)

with open('test_id.pickle', 'rb') as handle:
    test_id = pickle.load(handle)

# --- Lista de Partições ---
ROW_LIST = list(range(1, MAX_ROW_PARTITION + 1))
COL_LIST = list(range(1, MAX_COL_PARTITION + 1))
PARTITIONINGS = []
for r in ROW_LIST:
    for c in COL_LIST:
        if r == 1 and c == 1: continue
        PARTITIONINGS.append((r, c))

print(PARTITIONINGS)

# -----------------------------------------------------------------------------
# Avaliação do Modelo SVM BASE
# -----------------------------------------------------------------------------
print("\nAvaliando SVM Base Model...")
try:
    base_model = joblib.load('svm_base_model.joblib')

    y_pred_base = base_model.predict(X)

    true_part, total_part, pred_part = get_class_wise_accuracy_sklearn(y_one_hot, y_pred_base, prf=True)
    pre, rec, f1 = get_prf(true_part, total_part, pred_part)

    GLOBAL_MEAN = get_avg_f1(f1, total_part)

    fairness_loss_iter = get_fairness_loss_all_sklearn(y_one_hot, y_pred_base, PARTITIONINGS, test_id)
    base_fairness = np.sum(fairness_loss_iter)

    print("\n--- Resultados SVM BASE ---")
    print(f"F1 Global (Weighted): {GLOBAL_MEAN}")
    print(f"Métrica de Injustiça (Soma da Distância): {base_fairness}")

except FileNotFoundError:
    print("Erro: 'svm_base_model.joblib' não encontrado. Rode 'train_svm.py' primeiro.")

# -----------------------------------------------------------------------------
# Avaliação do Modelo SVM FINAL (SPAD)
# -----------------------------------------------------------------------------
print("\nAvaliando SVM Final (SPAD) Model...")
try:
    model = joblib.load('svm_final_model.joblib')

    y_pred_final = model.predict(X)

    true_part, total_part, pred_part = get_class_wise_accuracy_sklearn(y_one_hot, y_pred_final, prf=True)
    pre, rec, f1 = get_prf(true_part, total_part, pred_part)

    GLOBAL_MEAN = get_avg_f1(f1, total_part)

    fairness_loss_iter = get_fairness_loss_all_sklearn(y_one_hot, y_pred_final, PARTITIONINGS, test_id)
    final_fairness = np.sum(fairness_loss_iter)

    print("\n--- Resultados SVM FINAL (SPAD) ---")
    print(f"F1 Global (Weighted): {GLOBAL_MEAN}")
    print(f"Métrica de Injustiça (Soma da Distância): {final_fairness}")

except FileNotFoundError:
    print("Erro: 'svm_final_model.joblib' não encontrado. Rode 'train_svm.py' primeiro.")