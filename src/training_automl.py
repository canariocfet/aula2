from pycaret.regression import setup, compare_models, save_model, get_config
import pandas as pd
import numpy as np
import mlflow
import os

# === 1. Carregando os dados já limpos ===
sim_clean = np.load("outputs/sim_clean.npy")
df = pd.DataFrame(sim_clean, columns=["X", "Y", "Z", "Propriedade"])

# === 2. Configura o experimento no MLflow ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("projeto_aula4_mlops")

# === 3. Setup e AutoML ===
with mlflow.start_run(run_name="training_automl"):
    s = setup(
        data=df,
        target="Propriedade",
        session_id=123,
        log_experiment=True,
        experiment_name="projeto_aula4_mlops",
        log_plots=False,  # Evita tentar abrir gráficos fora do notebook
        verbose=False,
        html=False
    )

    best_model = compare_models(sort="RMSE")  # Você pode mudar para "MAE", "RMSE" etc

    # === 4. Salva modelo + dados de treino para etapa seguinte ===
    save_model(best_model, "outputs/best_model_pycaret")
    
    # Recupera os dados de treino/teste que PyCaret criou
    X_train = get_config("X_train")
    y_train = get_config("y_train")
    X_test = get_config("X_test")
    y_test = get_config("y_test")

    np.save("outputs/X.npy", pd.concat([X_train, X_test]).values)
    np.save("outputs/y.npy", pd.concat([y_train, y_test]).values)

    mlflow.sklearn.log_model(best_model, "modelo_pycaret")

print("Modelo AutoML salvo com sucesso.")
