import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path    
from Model_Development import predict_batch


def confusion_matrix(y_true, y_pred, pos):
    """
    Calcola la matrice di confusione binaria.

    Dati i vettori delle etichette reali e predette,
    restituisce il numero di veri positivi, falsi positivi,
    veri negativi e falsi negativi rispetto alla classe positiva.

    Parametri:
    - y_true: etichette reali
    - y_pred: etichette predette dal classificatore
    - pos: etichetta della classe positiva

    Ritorna:
    - TP, FP, TN, FN
    """
    TP = FP = TN = FN = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == pos and yp == pos:
            TP += 1
        elif yt != pos and yp == pos:
            FP += 1
        elif yt != pos and yp != pos:
            TN += 1
        else:
            FN += 1

    return TP, FP, TN, FN


def compute_metrics(TP, FP, TN, FN):
    """
    Calcola le principali metriche di valutazione per
    un problema di classificazione binaria.

    Le metriche restituite sono:
    - Accuracy
    - Error Rate
    - Sensitivity (Recall della classe positiva)
    - Specificity
    - G-Mean

    Parametri:
    - TP, FP, TN, FN: valori della matrice di confusione

    Ritorna:
    - tupla contenente tutte le metriche calcolate
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    gmean = (sensitivity * specificity) ** 0.5

    return accuracy, error_rate, sensitivity, specificity, gmean


def roc_curve_binary(y_true, y_pred, pos_label):
    """
    Costruisce i punti della curva ROC per un classificatore binario.

    Poiché il k-NN utilizzato restituisce solo classi discrete
    e non probabilità, la curva ROC viene approssimata utilizzando
    due soglie fisse.

    Parametri:
    - y_true: etichette reali
    - y_pred: etichette predette
    - pos_label: etichetta della classe positiva

    Ritorna:
    - lista di coppie (FPR, TPR)
    """
    thresholds = [1, 0]  # Soglie di decisione da usare per la curva ROC
    roc_points = []  # Lista dei punti (FPR, TPR) della ROC

    for t in thresholds:
        TP = FP = TN = FN = 0  # Inizializza i contatori della confusion matrix

        for yt, yp in zip(y_true, y_pred):  #
            pred_pos = 1 if yp == pos_label else 0  # Converte la predizione in binaria (0/1)

            if pred_pos >= t:
                if yt == pos_label:
                    TP += 1  # Vero positivo
                else:
                    FP += 1  # Falso positivo
            else:
                if yt == pos_label:
                    FN += 1  # Falso negativo
                else:
                    TN += 1  # Vero negativo

        TPR = TP / (TP + FN) if TP + FN > 0 else 0  # True Positive Rate (Recall)
        FPR = FP / (FP + TN) if FP + TN > 0 else 0  # False Positive Rate

        roc_points.append((FPR, TPR))  # Aggiunge il punto ROC per la soglia corrente

    roc_points.sort()  # Ordina i punti per FPR crescente
    return roc_points


def holdout(data, test_size, k, rng,positive_label):
    """
    Implementa il metodo di validazione holdout.

    Il dataset viene mescolato casualmente e diviso in un
    training set e un test set secondo la percentuale specificata.
    Il classificatore k-NN viene addestrato sul training set
    e valutato sul test set.

    Ritorna:
    - metriche di valutazione
    - matrice di confusione
    - etichette reali del test set
    - etichette predette dal classificatore
    """
    data = data.copy()  # Crea una copia del dataset per non modificare l’originale
    rng.shuffle(data)  # Mescola casualmente i dati

    split = int((1 - test_size) * len(data))  # Calcola l’indice di split train/test
    train = data[:split]  # Dati di training
    test = data[split:]  # Dati di test

    X_train, y_train = zip(*train)  # Separa feature ed etichette del training set
    X_test, y_test = zip(*test)  # Separa feature ed etichette del test set

    y_pred = predict_batch(  # Predice le etichette del test set con k-NN
        list(X_train),  # Feature di training
        list(y_train),  # Etichette di training
        list(X_test),  # Feature di test
        k,  # Numero di vicini
        rng #aggiungo seed
    )

    TP, FP, TN, FN = confusion_matrix(  # Calcola la confusion matrix
        y_test,  # Etichette reali
        y_pred,  # Etichette predette
        positive_label  # Classe positiva
    )

    metrics = compute_metrics(  # Calcola le metriche di valutazione
        TP, FP, TN, FN
    )

    return metrics, (TP, FP, TN, FN), y_test, y_pred  # Ritorna metriche, confusion matrix e risultati


def random_subsampling(data, test_size, k, n_experiments, rng,positive_label):
    """
    Implementa il metodo di validazione random subsampling.

    Il metodo holdout viene ripetuto più volte utilizzando
    split casuali differenti. Le metriche finali sono ottenute
    come media delle metriche calcolate in ogni esperimento.

    Ritorna:
    - media delle metriche di valutazione
    """
    results = []

    # Inizializzo i contatori globali della matrice di confusione
    TP_total = FP_total = TN_total = FN_total = 0

    # Liste globali per accumulare tutte le etichette reali e predette
    y_test_global = []
    y_pred_global = []

    # Ciclo su tutti gli esperimenti (random subsampling)
    for _ in range(n_experiments):
        # Eseguo un holdout per questo esperimento
        metrics, cm, y_test, y_pred = holdout(
            data,
            test_size,
            k,
            rng,
            positive_label
        )

        # Estrazione dei valori della matrice di confusione dell'esperimento corrente
        TP, FP, TN, FN = cm

        # Aggiornamento dei contatori globali sommando i risultati di questo esperimento
        TP_total += TP
        FP_total += FP
        TN_total += TN
        FN_total += FN

        # Accumulo delle etichette reali e predette per la ROC globale
        y_test_global.extend(y_test)
        y_pred_global.extend(y_pred)

        # Salvo le metriche dell'esperimento corrente per il calcolo della media
        results.append(metrics)

    # Calcolo della media delle metriche su tutti gli esperimenti
    metrics_mean = [
        sum(m[i] for m in results) / n_experiments
        for i in range(5)
    ]
    return (
        metrics_mean,
        (TP_total, FP_total, TN_total, FN_total),
        y_test_global,
        y_pred_global
    )

def stratified_cv(data, n_folds, k, rng, positive_label):
    """
    Implementa la stratified cross validation.

    I campioni vengono suddivisi in fold mantenendo
    la proporzione delle classi in ciascun fold.
    Ogni fold viene utilizzato una volta come test set.

    Ritorna:
    - media delle metriche di valutazione sui fold
    """
    class_groups = {}

    for x, y in data:
        class_groups.setdefault(y, []).append((x, y))

    folds = [[] for _ in range(n_folds)]

    for group in class_groups.values():
        random.shuffle(group)
        for i, sample in enumerate(group):
            folds[i % n_folds].append(sample)

    results = []
    # Inizializzo i contatori globali della matrice di confusione
    TP_total = FP_total = TN_total = FN_total = 0

    # Liste globali per accumulare tutte le etichette reali e predette
    y_test_global = []
    y_pred_global = []

    # Ciclo su tutti i fold (N_EXPERIMENTS = numero di fold)
    for i in range(n_folds):
        # Seleziono il fold i-esimo come test set
        test = folds[i]

        # Costruisco il training set unendo tutti gli altri fold
        train = [s for j, f in enumerate(folds) if j != i for s in f]

        # Separazione di feature e label per training e test
        X_train, y_train = zip(*train)
        X_test, y_test = zip(*test)

        # Predizione delle etichette del test set tramite k-NN
        y_pred = predict_batch(list(X_train), list(y_train), list(X_test), k,rng)

        # Calcolo della matrice di confusione per il fold corrente
        TP, FP, TN, FN = confusion_matrix(y_test, y_pred, positive_label)

        # Aggiornamento dei contatori globali sommando i valori di questo fold
        TP_total += TP
        FP_total += FP
        TN_total += TN
        FN_total += FN

        # Accumulo delle etichette reali e predette per la ROC globale
        y_test_global.extend(y_test)
        y_pred_global.extend(y_pred)

        # Calcolo le metriche per il fold corrente e le aggiungo alla lista results
        results.append(compute_metrics(TP, FP, TN, FN))

    # Calcolo la media delle metriche su tutti i fold
    metrics_mean = [
        sum(m[i] for m in results) / n_folds
        for i in range(5)
    ]

    return (
        metrics_mean,
        (TP_total, FP_total, TN_total, FN_total),
        y_test_global,
        y_pred_global
    )





# Stampa delle prestazioni del classificatore
def print_metrics(metrics):
    accuracy, error, sens, spec, gmean = metrics

    print("\n--- PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"G-Mean: {gmean:.4f}")


# Creazione del DataFrame contenente le metriche
def save_metrics_to_excel(metrics, output_path):
    """
    Salva le metriche di valutazione in un file Excel.

    Parametri:
    - metrics: tupla (accuracy, error, sensitivity, specificity, gmean)
    - output_path: percorso del file Excel
    """
    accuracy, error, sens, spec, gmean = metrics

    df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Error Rate": error,
        "Sensitivity": sens,
        "Specificity": spec,
        "G-Mean": gmean
    }])

    df.to_excel(output_path, index=False)



# Calcolo dei punti della ROC globale usando tutte le predizioni accumulate
def plot_roc_curve(y_true, y_pred, positive_label, output_path):
    """
    Calcola e salva la curva ROC per un classificatore binario.

    Parametri:
    - y_true: etichette reali
    - y_pred: etichette predette
    - positive_label: classe positiva
    - output_path: path immagine ROC
    """
    roc_points = roc_curve_binary(
        y_true,
        y_pred,
        positive_label
    )

    fpr = [p[0] for p in roc_points]
    tpr = [p[1] for p in roc_points]

    plt.figure()
    plt.plot(fpr, tpr, marker="o", label="k-NN")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - kNN")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Estrazione dei valori della matrice di confusione
def plot_confusion_matrix(cm,output_path, title="Confusion Matrix - kNN"):
    TP, FP, TN, FN = cm
    matrix = np.array([
        [TP, FP],
        [FN, TN]
    ])

    plt.figure()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Positive", "Predicted Negative"],
        yticklabels=["Actual Positive", "Actual Negative"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()