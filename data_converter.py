import numpy as np
import pandas as pd
import os

# Configuração baseada no seu projeto
# Você disse que INPUT_SIZE é 10 no código anterior. 
# Isso significa que as primeiras 10 colunas são X, o resto é y.
NUM_FEATURES = 10

def npy_to_csv(x_path, y_path, output_csv_name):
    """
    Lê X.npy e y.npy, concatena e salva como CSV.
    """
    print(f"--- Convertendo {x_path} e {y_path} para CSV ---")
    # 1. Carregar os arrays
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. {e}")
        return

    print(f"Shape original X: {X.shape}")
    print(f"Shape original y: {y.shape}")

    # 2. Verificar compatibilidade
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Erro: X tem {X.shape[0]} amostras, mas y tem {y.shape[0]}. Eles devem ser iguais.")

    # 3. Concatenar horizontalmente (lado a lado)
    # Se y for 1D (ex: [0, 1, 0]), transformamos para 2D antes
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    data_combined = np.hstack((X, y))

    # 4. Criar nomes para as colunas (facilita leitura)
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    label_cols = [f'label_{i}' for i in range(y.shape[1])]
    all_cols = feature_cols + label_cols

    # 5. Criar DataFrame e Salvar
    df = pd.DataFrame(data_combined, columns=all_cols)
    df.to_csv(output_csv_name, index=False)

    print(f"✅ Arquivo salvo com sucesso: {output_csv_name}")
    print(f"Dimensão final do CSV: {df.shape}")
    print("-" * 30)


def csv_to_npy(csv_path, x_output_path, y_output_path, num_features=NUM_FEATURES):
    """
    Lê um CSV, separa as primeiras 'num_features' como X e o resto como y,
    e salva de volta em .npy.
    """
    print(f"--- Convertendo {csv_path} de volta para NPY ---")
    # 1. Carregar CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    # 2. Separar X e y baseado no número de colunas de input
    # X pega todas as linhas (:), e colunas de 0 até num_features
    X_data = df.iloc[:, :num_features].values

    # y pega todas as linhas (:), e colunas de num_features até o final
    y_data = df.iloc[:, num_features:].values

    print(f"Recuperado X shape: {X_data.shape}")
    print(f"Recuperado y shape: {y_data.shape}")

    # 3. Salvar em .npy
    np.save(x_output_path, X_data.astype('float32')) # Garantindo float32 que é padrão do TF
    np.save(y_output_path, y_data.astype('float32')) # ou int dependendo do seu label, mas float é seguro p/ one-hot

    print(f"✅ Salvo X em: {x_output_path}")
    print(f"✅ Salvo y em: {y_output_path}")
    print("-" * 30)


# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # 1. GERAR CSVs (Para você aplicar sua privacidade)
    # ------------------------------------------------
    if os.path.exists('X_train.npy'):
        npy_to_csv('X_train.npy', 'y_train.npy', 'dados_treino.csv')

    if os.path.exists('X_test.npy'):
        npy_to_csv('X_test.npy', 'y_test.npy', 'dados_teste.csv')

    # ------------------------------------------------
    # 2. RECUPERAR NPY (EXEMPLO DE TESTE)
    # Use isso depois de aplicar seu script de privacidade no CSV.
    # Aqui vou simular convertendo o mesmo arquivo de volta só para testar.
    # ------------------------------------------------

    # Descomente as linhas abaixo quando quiser converter de volta:

    # csv_to_npy('dados_treino.csv', 'X_train_privado.npy', 'y_train_privado.npy')
    # csv_to_npy('dados_teste.csv', 'X_test_privado.npy', 'y_test_privado.npy')