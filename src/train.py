"""Treina um classificador usando dados sintéticos."""
from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=12, n_informative=7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
previsoes = modelo.predict(X_test)
print(f'Acurácia no teste: {accuracy_score(y_test, previsoes):.2%}')
print(classification_report(y_test, previsoes))
saida = Path(__file__).resolve().parents[1] / 'artifacts' / 'model.joblib'
saida.parent.mkdir(exist_ok=True)
joblib.dump(modelo, saida)
print(f'Modelo salvo em {saida}')
