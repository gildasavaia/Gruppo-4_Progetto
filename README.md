Descrizione del progetto per classificazione tumori con K-NN:
Questo progetto implementa una **pipeline completa di Machine Learning** per la classificazione di tumori come benigni (classe 2) o maligni (classe 4) a partire da caratteristiche numeriche.

L’obiettivo **non è massimizzare le prestazioni**, ma:
- costruire manualmente tutti i passaggi della pipeline
- rendere il codice il più possibile **generico e configurabile**
- permettere all’utente di **scegliere il metodo di validazione**
- **non utilizzare librerie di Machine Learning ad alto livello** (es. scikit-learn)

Il classificatore utilizzato è **k-Nearest Neighbors (k-NN)**, implementato interamente da zero.
Linguaggio: Python 3.8
Librerie utilizzate:
**- pandas**
**- numpy**
**- matplotlib**
**- seaborn**
**-argparse**
**os**
**random**
Installazione delle librerie attraverso il terminale: **pip install pandas numpy matplotlib seaborn argparse os random**

La pipeline è orchestrata da main.py ed è composta da 3 fasi principali:
**1. Preprocessing dei dati**
**2. Sviluppo del modello k-NN**
**3.Valutazione del modello**

Il **main.py** rappresenta il punto di ingresso dell’intera applicazione.
Responsabilità principali:
- parsing degli argomenti da linea di comando
- gestione dei percorsi input/output
- esecuzione del preprocessing
- scelta del metodo di validazione
- esecuzione del training e test
- generazione output e visualizzazion

Flusso operativo:
- caricamento dataset CSV
- preprocessing tramite data_preprocessing.py
- estrazione feature e label
- selezione metodo di validazione
- valutazione del modello k-NN
- salvataggio risultati
  
Output generati automaticamente:
- dataset preprocessato
- matrice di confusione
- curva ROC
- file Excel con metriche

Nel preprocessing dei dati abbiamo come funzione principale: **prepocessing(data)**.
Operazioni effettuate:
- selezione delle colonne rilevanti
- rimozione righe senza ID o classe
- pulizia valori numerici (virgola → punto)
- conversione feature a numerico
- imputazione valori mancanti tramite mediana
- filtraggio classi (solo 2 e 4)
- eliminazione righe con feature fuori dal range [1,10]

Sviluppo del modello e componenti principali:
- **distanza euclidea:** calcolo della distanza tra due vettori numerici.
- **calcolo distanze:** per ogni campione di test vengono calcolate le distanze da tutti i campioni di training.
- **selezione dei k vicini**: ordinamento per distanza crescente e selezione dei primi k. Il valore di k = 5 è stato scelto perché riduce la sensibilità al rumore rispetto a k = 1 e rappresenta un buon compromesso tra bias e varianza.   
- **predizione**: voto di maggioranza, ritorno causale in caso di parità controllata da seed.

Funzioni principali:
1. predict_one() → predizione singolo campione
2. predict_batch() → predizione batch

Input: nella cartella data version_1.csv

La valutazione del modello include: **metodi di validazione**, 
**- Holdout (H):**
1. Percentuale di dati destinata al training (es. 80%)
2. La parte restante viene utilizzata come test set.
**- Random Subsampling (B):**
 percentuale di training, numero di esperimenti indipendenti. Le metriche finali sono calcolate come media.
**Stratified Cross Validation (C):**
1. Numero di fold
2. Mantiene la distribuzione delle classi in ogni fold

L’utente deve selezionare il metodo di validazione tramite riga di comando: 
**Holdout:** python main.py --data-path Data/version_1.csv --validation holdout --train-perc 80 --k 5
**Random Subsampling**: python main.py --data-path Data/version_1.csv --validation subsampling --train-perc 70 --n-exp 20 --k 5
**Stratified Cross Validation**: python main.py --data-path Data/version_1.csv --validation cv --n-folds 10 --k 3

Per ogni esperimento vengono calcolate:
- Accuracy
- Error Rate
- Sensitivity (Recall della classe positiva)
- Specificity
- G-Mean

Le metriche vengono:
- visualizzate a schermo e salvate nel file **knn_results.xlsx**.

Output del Programma:
File generati vengono salvati nella cartella **Results**:
1. **knn_results.xlsx:** metriche di valutazione
2. **roc_curve_knn.png** curva ROC
3. **confusion_matrix_knn.png** matrice di confusione
4. **data_processed.csv**

Quello che si visualizza alla fine è una predizioni di classe:
2 → Benigno
4 → Maligno

Interpretazione dei risultati:
- Accuracy: accuratezza globale del classificatore
- Sensitivity: capacità di individuare tumori maligni
- Specificity: capacità di individuare tumori benigni
- G-Mean: equilibrio tra sensitivity e specificity

Curva ROC: Rappresenta il compromesso tra false Positive Rate,true Positive Rate

Matrice di Confusione: evidenzia
- Veri positivi (TP)
- Falsi positivi (FP)
- Veri negativi (TN)
- Falsi negativi (FN)

**Metriche e Visualizzazioni**
Oltre all’accuracy, sono state incluse metriche più adatte al contesto medico.
La curva ROC è approssimata, poiché il k-NN non fornisce probabilità.
La matrice di confusione permette una valutazione immediata degli errori.
Tutte le visualizzazioni vengono salvate come immagini per facilitare l’analisi.

Il progetto é stato completato con i file necessari per creare un'immagine docker dell'intera applicazione. 
Riga di codice utilizzata: **docker build -t <nome-immagine>.**
L'immagine é stata configurata in modo che un'istanza di container generata da essa:
- legga da una cartella della macchina ospite il dataset di partenza e scriva i risultati nella stessa cartella;

Per l'accesso al disco della macchina ospite da parte del container é stato utilizzato lo strumento del Bind Mounts -v 
riga di codice utilizzata: **docker run -it -v "path cartella":/app -w /app <nome-immagine> + uno degli esempi di esecuzione**.
