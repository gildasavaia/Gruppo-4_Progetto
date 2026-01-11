
from math import sqrt



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


def predict_one(X_train, y_train, x_test, k, rng):
    """
    Classifica un singolo campione di test.
    In caso di pareggio tra più classi, la classe predetta
    viene scelta casualmente tra quelle con voto massimo.
    """


    distances = compute_distances(X_train, y_train, x_test)
    neighbors = get_k_nearest(distances, k)

    class_votes = {}
    for _, label in neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1

    max_votes = max(class_votes.values())

    # classi che hanno il numero massimo di voti
    tied_classes = [
        label for label, votes in class_votes.items()
        if votes == max_votes
    ]

    # scelta casuale in caso di pareggio
    return rng.choice(tied_classes)

def predict_batch(X_train, y_train, X_test, k, rng):
    predictions = []
    for x_test in X_test:
        pred = predict_one(X_train, y_train, x_test, k, rng)
        predictions.append(pred)
    return predictions

