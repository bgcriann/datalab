import pandas as pd
import time

# ## Chargement du fichier de training
# Ici chargement le fichier data/train.csv et nettoyage :
# - conservation uniquement des colonnes d'intérêt
# - élimination des lignes avec des valeurs manquantes
# - numérisation de l'attribut 'Sex'
# - renommage des attributs (pour uniformisation avec l'exemple seaborn)

# TODO Etape 1 connexion au S3
from minio import Minio
client = Minio(endpoint=TODO,
               access_key=TODO,
               secret_key=TODO,
               secure=True)



#TODO Etape 2 read_csv sur S3
with client.get_object(
        bucket_name=TODO,
        object_name=TODO) as reader:
    titanic_train = pd.read_csv(reader)
print(titanic_train.head())

# Ici nettoyage :
# - conservation uniquement des colonnes d'intérêt
# - élimination des lignes avec des valeurs manquantes
# - numérisation de l'attribut 'sex'

titanic_train = titanic_train[['Survived','Pclass','Sex','Age']]
titanic_train.dropna(axis=0, inplace=True)
titanic_train['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
# Renommage de colonnes (homogene seaborn)
column_name = {'PassengerId':'passengerid',
               'Survived':'survived',
               'Pclass':'pclass',
               'Sex':'sex',
               'Age':'age'}
titanic_train.rename(columns=column_name, inplace=True)
print(titanic_train.head())


# # Entrainement du modèle
# ## Sélection de la cible et des features d'entraînement
y_train = titanic_train['survived']
X_train = titanic_train.drop('survived', axis=1)


# ## Entraînement sur un modèle KNN
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
t_start = time.process_time()
model.fit(X_train, y_train) # entrainement du modele
t_stop = time.process_time()
print(f'score du model KNN {model.score(X_train, y_train):.2f} obtenu en {t_stop - t_start :.4f} seconde(s)') # évaluation


# ## Sauvegarde du modèle dans un fichier binaire
import pickle
with open('model.sav','wb') as writer:
    pickle.dump(model, writer)

# ## TODO Sauvegarde du fichier sur le s3
client.fput_object(bucket_name=TODO,
                   object_name=TODO,
                   file_path=TODO)
