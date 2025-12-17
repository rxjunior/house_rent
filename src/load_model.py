import mlflow.sklearn
import pandas as pd
import os

# 1. Configurar o MLflow para encontrar o modelo
# Como este script será executado da raiz do projeto, o URI é o caminho local.
MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_latest_model(model_name: str):
    """
    Carrega a última versão registrada de um modelo específico do MLflow.
    """
    try:
        # A sintaxe 'models:/' é usada para carregar modelos registrados.
        model_uri = f"models:/{model_name}/latest"
        
        # O MLflow carrega o pipeline completo (ColumnTransformer + Regressor)
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        print(f"Modelo '{model_name}' (última versão) carregado com sucesso do MLflow.")
        return loaded_model
        
    except Exception as e:
        print(f"Erro ao carregar o modelo '{model_name}': {e}")
        # Se o modelo não estiver registrado, ele retorna None
        return None

# Definindo o nome do modelo vencedor (use o nome que obteve o melhor RMSE)
MODEL_NAME = "RandomForest" # Mude para "XGBoost" ou "RidgeRegression" se foi o melhor.

# Carregar o modelo assim que o módulo for importado
MODEL = load_latest_model(MODEL_NAME)