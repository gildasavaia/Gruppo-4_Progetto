import random

from Model_Development import predict_batch, X, y # importiamo le funzioni e i dati dal file di sviluppo del modello

data_xy = list(zip(X, y)) # uniamo caratteristiche e etichette i ua lista di tuple(X, y)

random.seed(42)          # per rendere riproducibile lo split
random.shuffle(data_xy) # mescoliamo l'ordine dei campioni del dataset 

split_index = int(0.8 * len(data_xy)) 
"""
calcolo l’indice di split per usare l’80% dei dati come training set
ed il restante 20% sarà il test set

"""

train_data = data_xy[:split_index] # prendiamo l'80% dei dati per il training set
test_data = data_xy[split_index:] # prendiamo il restante 20% per il test set

X_train, y_train = zip(*train_data) #separiamo caratteristiche e etichette del training set
X_test, y_test = zip(*test_data) # separiamo caratteristiche e etichette del test set

X_train = list(X_train)
y_train = list(y_train)
X_test = list(X_test)
y_test = list(y_test)

"""
Ora abbiamo:
- X_train, y_train: dati di addestramento
- X_test, y_test: dati di test

"""

#esecuzione del classificatore knn
k = 5  #numero di vicini più prossimi da considerare 

#applichiamo il classificatore knn a tutti i campioni del test set e stampiamo le predizioni sul test set
y_pred = predict_batch(X_train, y_train, X_test, k)
print("Predictions:", y_pred)

