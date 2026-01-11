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
- pandas
- numpy
- matplotlib
- seaborn

Installazione delle librerie attraverso il terminale: pip install pandas numpy matplotlib seaborn

Pipeline del progetto: la pipeline è composta da tre fasi principali:
***1. Preprocessing dei dati***
***2. Sviluppo del modello k-NN**
***3.Valutazione del modello***

Ogni fase è implementata in modo modulare e indipendente.

1. Preprocessing dei dati
   
**Input:**
- File CSV contenente il dataset originale (version_1.csv)
- Operazioni eseguite
- Selezione delle colonne rilevanti
- Rimozione delle righe prive di identificativo o classe
- Conversione dei valori da stringa a numerico
- Gestione dei valori mancanti tramite mediana
- Filtraggio delle classi (solo 2 e 4)
- Eliminazione delle righe con feature fuori dal range [1,10]
**Output:**
- File CSV preprocessato (dataset_preprocessato.csv)
- Dataset pronto per l’addestramento e la valutazione
**Output di Controllo**
- Durante l’esecuzione del programma vengono stampati a schermo:
- Dimensioni della matrice delle feature X
- Dimensioni del vettore delle etichette y
- Distribuzione delle classi nel dataset

2. Sviluppo del Modello k-NN

Modello utilizzato:
- Algoritmo: k-Nearest Neighbors (k-NN)
- Distanza: Euclidea
- Strategia di classificazione: Voto di maggioranza

Funzionalità implementate:
- Calcolo della distanza euclidea
- Calcolo delle distanze tra un campione di test e il training set
- Selezione dei k vicini più prossimi
- Predizione della classe per un singolo campione
- Predizione batch sull’intero test set

Input: Dataset preprocessato

Output: predizioni di classe:
2 → Benigno
4 → Maligno

Scelta del Parametro k: il valore di k = 5 è stato scelto perché
- riduce la sensibilità al rumore rispetto a k = 1
- rappresenta un buon compromesso tra bias e varianza
Il valore di k può essere facilmente modificato nel codice.

3. Valutazione del Modello:
Questo script rappresenta il punto di ingresso principale per l’utente.
Esecuzione:
python Model_Evaluation.py

Scelte di Input dell’Utente: all’avvio del programma, l’utente deve selezionare il metodo di validazione:
- Holdout (H) : 1. Percentuale di dati destinata al training (es. 80%)
				2. La parte restante viene utilizzata come test set

- Random Subsampling (B): Percentuale di training, Numero di esperimenti indipendenti.
						  Le metriche finali sono calcolate come media.

- Stratified Cross Validation (C): 1. Numero di fold
								   2.Mantiene la distribuzione delle classi in ogni fold
								   3. Metriche di Valutazione
Per ogni esperimento vengono calcolate:
- Accuracy
- Error Rate
- Sensitivity (Recall della classe positiva)
- Specificity
- G-Mean

Le metriche vengono:
- visualizzate a schermo e salvate nel file knn_results.xlsx

Output del Programma:
File generati

1. knn_results.xlsx → metriche di valutazione
2. roc_curve_knn.png → curva ROC
3. confusion_matrix_knn.png → matrice di confusione

4. Interpretazione dei Risultati:
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

Metriche e Visualizzazioni
Oltre all’accuracy, sono state incluse metriche più adatte al contesto medico
La curva ROC è approssimata, poiché il k-NN non fornisce probabilità
La matrice di confusione permette una valutazione immediata degli errori
Tutte le visualizzazioni vengono salvate come immagini per facilitare l’analisi.

Il progetto é stato completato con i file necessari per creare un'immagine
docker dell'intera applicazione. 

Riga di codice utilizzata: docker build -t <nome-immagine>.

L'immagine é stata configurata in modo che un'istanza dicontainer generata da essa:
- legga da una cartella della macchina ospite il dataset di partenza e scriva i risultati nella stessa cartella;

Per l'accesso al disco della macchina ospite da parte del container é stato utilizzato lo strumento del Bind Mounts -v 

riga di codice utilizzata: docker run -it -v C:/Users/Gabriele/Desktop/Progetto_gruppo-finito:/app -w /app <nome-immagine> python Model_Evaluation.py



