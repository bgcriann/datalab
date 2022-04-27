import seaborn as sns
import time

# # Chargement et nettoyage des données

# ## chargement du dataset via seaborn

titanic = sns.load_dataset('titanic')
print(titanic.head())


# Ici nettoyage :
# - conservation uniquement des colonnes d'intérêt
# - élimination des lignes avec des valeurs manquantes
# - numérisation de l'attribut 'sex'

titanic = titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male', 'female'], [0, 1], inplace=True)
print(titanic.head())


# # Entrainement du modèle

# ## Sélection de la cible et des features d'entraînement
y_train = titanic['survived']
X_train = titanic.drop('survived', axis=1)


# ## Entraînement sur un modèle KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
t_start = time.process_time()
model.fit(X_train, y_train) # entrainement du modele
t_stop = time.process_time()
print(f'score du model KNN {model.score(X_train, y_train):.2f} obtenu en {t_stop - t_start :.4f} seconde(s)') # évaluation

