import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données depuis le fichier Excel
df = pd.read_excel('C:/Users/hp/OneDrive/Bureau/Nouveau dossier (2)/station dernier.xlsx', engine='openpyxl')

# Afficher les premières lignes pour vérifier
print(df.head())

# Supprimer les colonnes non numériques, comme "Mois"
df = df.select_dtypes(include=['float64', 'int64'])

# Sélectionner les caractéristiques (X) et la cible (y)
X = df.drop(columns=['Stock Fin (L)'])  # Utiliser toutes les colonnes sauf 'Stock Fin (L)'
y = df['Stock Fin (L)']  # La colonne cible

# Diviser les données en 70% pour l'entraînement et 30% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédire les valeurs pour les données de test
y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE) pour évaluer le modèle
mse = mean_squared_error(y_test, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse}")

# Afficher les coefficients de la régression
print("Coefficients : ", model.coef_)
print("Intercept : ", model.intercept_)
import joblib



import os
print("Répertoire de travail courant :", os.getcwd())

# Sauvegarder ton modèle dans un fichier .joblib
joblib.dump(model, 'modele_regr_lineaire.joblib')

# Cela va créer un fichier "modele_regr_lineaire.joblib" dans ton répertoire de travail
