import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Model_Development1 import predict_batch, X, y

# Numero di vicini utilizzati dal classificatore k-NN
K_NEIGHBORS = 5
# Imposto il seed per rendere i risultati riproducibili
random.seed(42)
POSITIVE_LABEL = 4


print("Seleziona il metodo di validazione:")
print("H -> Holdout")
print("B -> Random Subsampling")
print("C -> Stratified Cross Validation")

choice = input("Inserisci la tua scelta (H/B/C): ").strip().upper()

if choice == "H":
    VALIDATION = "holdout"

    train_perc = float(
        input("Inserisci la percentuale di dati per il training set (es. 80): ")
    )

    TEST_SIZE = 1 - train_perc / 100

elif choice == "B":
    VALIDATION = "B"

    train_perc = float(
        input("Inserisci la percentuale di dati per il training set (es. 80): ")
    )
    TEST_SIZE = 1 - train_perc / 100

    N_EXPERIMENTS = int(
        input("Inserisci il numero di esperimenti per il random subsampling: ")
    )

elif choice == "C":
    VALIDATION = "C"

    N_EXPERIMENTS = int(
        input("Inserisci il numero di fold per la cross validation stratificata: ")
    )

else:
    raise ValueError("Scelta non valida. Inserire H, B o C.")
# Unisco le feature e le etichette in una lista di coppie (x, y)
# per facilitare shuffle e split del dataset
data_xy = list(zip(X, y))


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


def holdout(data):
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
    random.shuffle(data)  # Mescola casualmente i dati

    split = int((1 - TEST_SIZE) * len(data))  # Calcola l’indice di split train/test
    train = data[:split]  # Dati di training
    test = data[split:]  # Dati di test

    X_train, y_train = zip(*train)  # Separa feature ed etichette del training set
    X_test, y_test = zip(*test)  # Separa feature ed etichette del test set

    y_pred = predict_batch(  # Predice le etichette del test set con k-NN
        list(X_train),  # Feature di training
        list(y_train),  # Etichette di training
        list(X_test),  # Feature di test
        K_NEIGHBORS  # Numero di vicini
    )

    TP, FP, TN, FN = confusion_matrix(  # Calcola la confusion matrix
        y_test,  # Etichette reali
        y_pred,  # Etichette predette
        POSITIVE_LABEL  # Classe positiva
    )

    metrics = compute_metrics(  # Calcola le metriche di valutazione
        TP, FP, TN, FN
    )

    return metrics, (TP, FP, TN, FN), y_test, y_pred  # Ritorna metriche, confusion matrix e risultati


def random_subsampling(data):
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
    for _ in range(N_EXPERIMENTS):
        # Eseguo un holdout per questo esperimento
        metrics, cm, y_test, y_pred = holdout(data)

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
    metrics_mean = [sum(m[i] for m in results) / N_EXPERIMENTS for i in range(5)]
    return metrics_mean, (TP_total, FP_total, TN_total, FN_total), y_test_global, y_pred_global

def stratified_cv(data):
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

    folds = [[] for _ in range(N_EXPERIMENTS)]

    for group in class_groups.values():
        random.shuffle(group)
        for i, sample in enumerate(group):
            folds[i % N_EXPERIMENTS].append(sample)

    results = []
    # Inizializzo i contatori globali della matrice di confusione
    TP_total = FP_total = TN_total = FN_total = 0

    # Liste globali per accumulare tutte le etichette reali e predette
    y_test_global = []
    y_pred_global = []

    # Ciclo su tutti i fold (N_EXPERIMENTS = numero di fold)
    for i in range(N_EXPERIMENTS):
        # Seleziono il fold i-esimo come test set
        test = folds[i]

        # Costruisco il training set unendo tutti gli altri fold
        train = [s for j, f in enumerate(folds) if j != i for s in f]

        # Separazione di feature e label per training e test
        X_train, y_train = zip(*train)
        X_test, y_test = zip(*test)

        # Predizione delle etichette del test set tramite k-NN
        y_pred = predict_batch(list(X_train), list(y_train), list(X_test), K_NEIGHBORS)

        # Calcolo della matrice di confusione per il fold corrente
        TP, FP, TN, FN = confusion_matrix(y_test, y_pred, POSITIVE_LABEL)

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
    metrics_mean = [sum(m[i] for m in results) / N_EXPERIMENTS for i in range(5)]
    return metrics_mean, (TP_total, FP_total, TN_total, FN_total), y_test_global, y_pred_global


# Selezione del metodo di validazione scelto
if VALIDATION == "holdout":
    metrics, cm, y_test, y_pred = holdout(data_xy)
elif VALIDATION == "B":
    metrics, cm, y_test, y_pred = random_subsampling(data_xy)
elif VALIDATION == "C":
    metrics, cm, y_test, y_pred = stratified_cv(data_xy)
else:
    raise ValueError("Metodo di validazione non valido")
# Estrazione delle metriche di valutazione
accuracy, error, sens, spec, gmean = metrics

# Stampa delle prestazioni del classificatore
print("\n--- PERFORMANCE ---")
print("Accuracy:", accuracy)
print("Error Rate:", error)
print("Sensitivity:", sens)
print("Specificity:", spec)
print("G-Mean:", gmean)

# Creazione del DataFrame contenente le metriche
df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Error Rate": error,
    "Sensitivity": sens,
    "Specificity": spec,
    "G-Mean": gmean
}])

# Salvataggio delle metriche in un file Excel
df.to_excel("knn_results.xlsx", index=False)

# Visualizzazione dei grafici solo nel caso di validazione holdout

# Calcolo dei punti della ROC globale usando tutte le predizioni accumulate
roc_points = roc_curve_binary(y_test, y_pred, POSITIVE_LABEL)  # calcolo FPR e TPR
fpr = [p[0] for p in roc_points]  # estraggo i valori di False Positive Rate
tpr = [p[1] for p in roc_points]  # estraggo i valori di True Positive Rate


# Plot della curva ROC
plt.figure()
plt.plot(fpr, tpr, marker="o", label="k-NN")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - kNN")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_knn.png", dpi=300, bbox_inches="tight")
plt.show()

# Estrazione dei valori della matrice di confusione
TP, FP, TN, FN = cm
matrix = [[TP, FP], [FN, TN]]

# Visualizzazione della matrice di confusione
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
plt.title("Confusion Matrix - kNN")
plt.savefig("confusion_matrix_knn.png", dpi=300, bbox_inches="tight")
plt.show()
