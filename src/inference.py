<<<<<<< HEAD
### Modulo 4. inference.py ➞ aplica inferência no seismic_slice

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula2_modulo_mlops as sts
import mlflow
import joblib

#Carregando os dados para a inferencia
seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")
X = np.load("outputs/X.npy")
y = np.load("outputs/y.npy")

#Carregando o modelo treinado
ET = joblib.load("outputs/model.pkl")
sts.ET = ET

#### Aplica o modelo ML treinado e faz a inferencia
seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice_clean, modelo = ET, X=X, y=y)

np.save("outputs/seis_estimated.npy", seis_estimated)

print("Inferencia Pronta")

#### FIM
=======
### inference.py – aplica o modelo treinado do PyCaret aos dados sísmicos

import numpy as np
import pandas as pd
from pycaret.regression import load_model, predict_model
import mlflow

# === 1. Carregando dados sísmicos limpos ===
seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")

# Convertendo para DataFrame com nomes coerentes
df_seismic = pd.DataFrame(seismic_slice_clean, columns=["X", "Y", "Z", "Propriedade"])

# === 2. Carregando modelo treinado pelo PyCaret ===
model = load_model("outputs/best_model_pycaret")

# === 3. Inferência ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("projeto_aula4_mlops")

with mlflow.start_run(run_name="inference_automl"):
    # Primeiro faz a predição
    predictions = predict_model(model, data=df_seismic)

    # Renomeia a coluna de predição
    predictions = predictions.rename(columns={"prediction_label": "Propriedade_Prevista"})

    # Inclui coordenadas + previsão como array 2D
    seis_estimated = predictions[["X", "Y", "Z", "Propriedade_Prevista"]].values
    np.save("outputs/seis_estimated.npy", seis_estimated)

    # Opcional: salvar o DataFrame inteiro com coords + previsões
    predictions.to_csv("outputs/seismic_predictions.csv", index=False)

print("Inferência com PyCaret concluída.")
>>>>>>> 2950874 (Versão com AutoML PyCaret + DVC + MLflow funcionando)
