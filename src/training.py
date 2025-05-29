### Modulo 3. training.py ➞ treina o modelo ExtraTrees

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
<<<<<<< HEAD
from modulo import aula2_modulo_mlops as sts
=======
from modulo import aula3_modulo_mlops as sts
>>>>>>> 2950874 (Versão com AutoML PyCaret + DVC + MLflow funcionando)
import mlflow
import joblib

# Carrega dados de treino
sim_clean = np.load("outputs/sim_clean.npy")

#Treinandoo modelo nos dados de treino
sim_estimado, y, nrms_teste, r2_teste, mape_teste, dict_params, ET, X = sts.ML_model_evaluation(dados_simulacao=sim_clean, proporcao_treino=0.75)

#Exportando o modelo treinado
joblib.dump(ET, "outputs/model.pkl")

<<<<<<< HEAD
np.save("outputs/X.npy",X)
np.save("outputs/y.npy",y)

=======
#Exportando os dados para a inferencia
np.save("outputs/X.npy",X)
np.save("outputs/y.npy",y)

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Garante que aponta para o local correto da URL
mlflow.set_experiment("projeto_aula4_mlops") #

with mlflow.start_run(run_name="training"):
    mlflow.log_params(dict_params)
    mlflow.log_param("modelo", "ExtraTrees")
    mlflow.log_metrics({
                        "nrms_teste": nrms_teste,
                        "r2_teste": r2_teste,
                        "mape_teste": mape_teste
                        })
    mlflow.sklearn.log_model(ET, "modelo_extra_trees")

>>>>>>> 2950874 (Versão com AutoML PyCaret + DVC + MLflow funcionando)
print("Treino do ML pronto")

### FIM
