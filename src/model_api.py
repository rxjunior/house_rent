from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Importa o modelo carregado e o nome do modelo do nosso script auxiliar
from load_model import MODEL, MODEL_NAME

# --- 1. Inicialização do FastAPI ---
app = FastAPI(
    title="House Prices Prediction API",
    description="API para prever preços de casas usando o modelo de regressão treinado.",
    version="1.0.0"
)

# --- 2. Definição do Esquema de Dados (Pydantic) ---
# O Pydantic garante que a requisição JSON tenha as colunas esperadas.
# Apenas listamos algumas colunas cruciais aqui.
class HouseFeatures(BaseModel):
    LotArea: int = 8450
    OverallQual: int = 7
    OverallCond: int = 5
    YearBuilt: int = 2003
    Neighborhood: str = "CollgCr"
    
    # É fundamental que o schema tenha as colunas que o modelo espera,
    # mesmo que usemos apenas algumas para o exemplo.
    # Para o seu projeto real, você listaria *todas* as 79 colunas do dataset.


# --- 3. Rotas da API ---

@app.get("/")
def health_check():
    """Verifica se a API está rodando e qual modelo está sendo usado."""
    if MODEL:
        return {"status": "ok", "model_loaded": MODEL_NAME}
    else:
        # Erro 503: Serviço Indisponível (modelo não carregado)
        raise HTTPException(status_code=503, detail="Modelo não carregado. Verifique os logs do MLflow.")

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """Recebe dados de entrada e retorna a previsão de preço."""
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível para previsão.")

    try:
        # 1. Converte o objeto Pydantic em DataFrame
        # A previsão do Scikit-learn (e o Pipeline) espera um DataFrame,
        # com os nomes exatos das colunas brutas.
        input_data = features.dict()
        
        # FastAPI precisa de uma linha por vez, mas o modelo precisa de um DataFrame
        data_df = pd.DataFrame([input_data])
        
        # 2. Faz a previsão
        # O Pipeline aplica o ColumnTransformer e o Regressor em um só passo
        prediction = MODEL.predict(data_df)
        
        # 3. Formata e retorna o resultado
        predicted_price = float(prediction[0])
        
        return {
            "predicted_sale_price": round(predicted_price, 2),
            "model_used": MODEL_NAME
        }
        
    except Exception as e:
        # Loga o erro interno (opcional, mas recomendado em produção)
        print(f"Erro na previsão: {e}")
        # Erro 500: Erro Interno do Servidor (problema no processamento do dado)
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a requisição. Detalhe: {e}")