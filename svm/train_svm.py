import numpy as np
import pickle
import random
import time
import joblib
from sklearn.linear_model import SGDClassifier
from keras import backend as K

from svm_utils import get_class_wise_accuracy_sklearn, get_prf, get_avg_f1, get_weighted_f1_one_sklearn

# --- Global Variable ---
MAX_ROW_PARTITION = 5
MAX_COL_PARTITION = 5

print("Carregando dados de treino...")
X = np.load('X_train.npy')
y_one_hot = np.load('y_train.npy')

y_labels = np.argmax(y_one_hot, axis=1)
all_classes = np.unique(y_labels)

with open('train_id.pickle', 'rb') as handle:
    train_id = pickle.load(handle)

base_model = SGDClassifier(loss='log_loss', random_state=42)

print("Treinando o modelo SVM Base...")
base_model.fit(X, y_labels)

joblib.dump(base_model, 'svm_base_model.joblib')
print("Modelo SVM Base salvo em 'svm_base_model.joblib'")

print("Avaliando o modelo base nos dados de treino...")
y_pred_labels = base_model.predict(X)
true_part, total_part, pred_part = get_class_wise_accuracy_sklearn(y_one_hot, y_pred_labels, prf=True)
pre, rec, f1 = get_prf(true_part, total_part, pred_part)

GLOBAL_MEAN = get_avg_f1(f1, total_part)
print(f"SVM Base - F1 Global (Treino): {GLOBAL_MEAN}")

def get_partition_data(index1, index2, X_data, y_data, all_partitioning_data_list):
    """
    Retorna dados particionados (labels 1D para sklearn).
    """
    X_train_parts = []
    y_train_parts = []

    data_list = all_partitioning_data_list[(index1, index2)]
    for i in range(index1 * index2):
        indices = data_list[i]
        X_train_parts.append(X_data[indices])
        y_train_parts.append(y_data[indices])

    return X_train_parts, y_train_parts

def set_sample_weights(index1, index2, w_f1):
    """
    Mesma lógica do set_lr_weight, mas retorna pesos de amostra.
    """
    global sample_weights_list, GLOBAL_MEAN

    sample_weights_list = np.zeros((index1 * index2))
    sample_weights_list = (GLOBAL_MEAN - w_f1)
    
    print(f"Desvio de F1 (alvo - atual): {sample_weights_list}")
    
    lambda_value = 0.0005 # Valor do artigo original

    sample_weights_list = np.maximum(sample_weights_list, 0) # Equivalente ao tf.nn.relu
    
    if np.sum(sample_weights_list) > 0:
        # Normaliza os pesos (esta lógica é do artigo)
        sample_weights_list = sample_weights_list / np.sum(sample_weights_list) * lambda_value * (index1 * index2)
        sample_weights_list = sample_weights_list / (np.amax(sample_weights_list) / lambda_value)

    sample_weights_list += K.epsilon() # Evita peso zero
    return sample_weights_list

# Define partições
ROW_LIST = list(range(1, MAX_ROW_PARTITION + 1))
COL_LIST = list(range(1, MAX_COL_PARTITION + 1))
PARTITIONINGS = []
for r in ROW_LIST:
    for c in COL_LIST:
        if r == 1 and c == 1: continue
        PARTITIONINGS.append((r, c))

print(f"Total de partições a testar: {len(PARTITIONINGS)}")


# Treinamento Bi-level (Justiça) (SPAD)
print("\nIniciando Treinamento de Justiça (Bi-level) para SVM...")
start_time = time.time()

model = base_model

loop_list = [4, 30] # [2, 5] Caso queira rodar mais rápido descomentar
epoch_list = [5, 1] # [2, 1] Caso queira rodar mais rápido descomentar


for i in range(len(loop_list)):
    epochs = epoch_list[i]
    loops = loop_list[i]
    print(f"\n--- Fase de Treino {i+1}: {loops} loops, {epochs} épocas por partição ---")

    for l in range(loops):
        random.shuffle(PARTITIONINGS)

        for j in range(len(PARTITIONINGS)):
            (index1, index2) = PARTITIONINGS[j]

            X_train_parts, y_train_parts = get_partition_data(index1, index2, X, y_labels, train_id)

            print(f"Loop {l+1}/{loops}, Partição {j+1}/{len(PARTITIONINGS)}: ({index1}, {index2})")

            w_f1_scores = get_weighted_f1_one_sklearn(model, X, y_one_hot, train_id, index1, index2)
            partition_weights = set_sample_weights(index1, index2, w_f1_scores)

            for e in range(epochs):
                for p in range(index1 * index2):
                    X_p = X_train_parts[p]
                    y_p = y_train_parts[p]

                    if len(y_p) == 0: continue

                    current_weight = partition_weights[p]
                    sample_weights_for_batch = np.full(shape=len(y_p), fill_value=current_weight, dtype=float)

                    model.partial_fit(X_p, y_p, sample_weight=sample_weights_for_batch, classes=all_classes)

print("Treinamento de Justiça concluído.")
joblib.dump(model, 'svm_final_model.joblib')
print("Modelo SVM Final salvo em 'svm_final_model.joblib'")
print("Tempo total: %f s" % (time.time() - start_time))