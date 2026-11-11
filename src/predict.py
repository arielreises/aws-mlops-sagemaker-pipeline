"""Carrega o modelo treinado e faz uma previsão de exemplo."""
from pathlib import Path
import joblib
import numpy as np

arquivo = Path(__file__).resolve().parents[1] / 'artifacts' / 'model.joblib'
if not arquivo.exists():
    raise SystemExit('Execute primeiro: python src/train.py')
modelo = joblib.load(arquivo)  # Carregar apenas artefatos locais confiáveis.
amostra = np.zeros((1, modelo.n_features_in_))
print('Classe prevista:', int(modelo.predict(amostra)[0]))
