import numpy as np

files_to_check = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']

print("Verificando dataset")

for f in files_to_check:
    try:
        data = np.load(f)
        dimensions = data.shape

        print(f"\n Arquivo: {f}")
        print(f" Dimensões: {dimensions}")

        if len(dimensions) == 2:
            print(f" {dimensions[0]:,} amostras (linhas)")
            print(f" {dimensions[1]:,} features (colunas)")
        else:
             print(f" {dimensions[0]:,} amostras (linhas)")

    except FileNotFoundError:
        print(f"\nArquivo: {f}")
        print(f" Erro: Arquivo não encontrado.")

    except Exception as e:
        print(f"\nArquivo: {f}")
        print(f" Erro ao ler o arquivo: {e}")

print("\n Finalizado.")