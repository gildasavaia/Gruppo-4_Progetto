import data_preprocessing as dp
import random
from math import sqrt
import pandas as pd

# Carica il dataset preprocessato
df = pd.read_csv("dataset_preprocessato.csv")
X, y = dp.get_features_and_labels(df) #abbiamo estratto X e y dal dataframe

#distaza euclidea

def euclidean_distance(a, b):
    """
    Calcola la distanza euclidea tra due vettori a e b
    """
    s = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        s += diff * diff
    return sqrt(s)

#calcolo distaza da tutto il training set

def compute_distances(X_train, y_train, x_test):
    """
    Per un campione di test:
    - calcola la distanza da TUTTI i campioni di training
    - restituisce una lista di (distanza, label)
    """
    distances = []
    for x_tr, y_tr in zip(X_train, y_train):
        d = euclidean_distance(x_test, x_tr)
        distances.append((d, y_tr))
    return distances

#selezione dei k vicini più prossimi


def get_k_nearest(distances, k):
    """
    Ordina per distanza crescente e seleziona i primi k
    """
    distances.sort(key=lambda t: t[0])
    return distances[:k]

#predizione della classe tramite voto di maggioranza
def predict_one(X_train, y_train, x_test, k):
    """
    Classifica un singolo campione di test
    """
    distances = compute_distances(X_train, y_train, x_test)
    neighbors = get_k_nearest(distances, k)
    class_votes = {} #conteggio delle label 

    for _, label in neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1

    # Restituisce la classe con il maggior numero di voti
    return max(class_votes, key=class_votes.get)

def predict_batch(X_train, y_train, X_test, k):
    #applico knn a tutto il test set
    predictions = []
    for x_test in X_test:
        pred = predict_one(X_train, y_train, x_test, k)
        predictions.append(pred)
    return predictions

