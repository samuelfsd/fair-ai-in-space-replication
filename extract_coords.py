import pickle
import numpy as np
import pandas as pd

MAP_WIDTH = 4096
MAP_HEIGHT = 4096

def extract_to_csv(pickle_path, output_csv):
    print(f"--- Lendo {pickle_path} ---")
    with open(pickle_path, 'rb') as handle:
        train_id = pickle.load(handle)

    # 1. Coletar todos os índices únicos usados no treino
    any_key = list(train_id.keys())[0]
    partitions = train_id[any_key]
    print("Extraindo e achatando arrays...")

    # CORREÇÃO AQUI:
    # Em vez de usar lista e set, usamos NumPy para garantir que arrays sejam tratados corretamente
    # 1. Convertemos cada partição para array e usamos .ravel() para garantir que seja 1D (achatado)
    # 2. Concatenamos tudo num array gigante
    flat_indices = np.concatenate([np.array(p).ravel() for p in partitions])

    # 3. Usamos np.unique que faz o trabalho do set() mas funciona com arrays
    unique_indices = np.unique(flat_indices)

    print(f"Total de amostras de treino encontradas: {len(unique_indices)}")

    # 2. Converter Índice Linear -> Coordenadas (X, Y)
    # Fórmula: x = index % width, y = index // width
    x_coords = unique_indices % MAP_WIDTH
    y_coords = unique_indices // MAP_WIDTH

    # 3. Criar DataFrame e Salvar
    df = pd.DataFrame({
        'id_amostra': unique_indices, # Esse ID conecta com a linha do X_train.npy
        'x': x_coords,
        'y': y_coords
    })

    df.to_csv(output_csv, index=False)
    print(f"✅ Arquivo salvo: {output_csv}")
    print("Envie este arquivo para seu colega aplicar o Geo-Indistinguishability.")

if __name__ == "__main__":
    extract_to_csv('train_id.pickle', 'localizacoes_originais.csv')