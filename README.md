# Gruppo-4_Progetto
Classificazione di Tumori tramite Machine Learning
Step 1 – Preprocessing del Dataset
Descrizione del progetto:
Questo progetto ha come obiettivo la realizzazione di una pipeline di apprendimento automatico per la classificazione di tumori come benigni o maligni, a partire da un insieme di caratteristiche cellulari.

Il lavoro è strutturato in più fasi.
Nella prima fase viene implementato esclusivamente il preprocessing del dataset, necessario per garantire la qualità e la coerenza dei dati prima dell’addestramento dei modelli. Dataset che viene elaborato e modificato così da evitare errori.
Il dataset fornito contiene informazioni cliniche relative a cellule tumorali ed è composto da:
- ID: identificativo del campione (Sample code number)
- 9 feature numeriche, ciascuna con valori compresi tra 1 e 10
- Label di classe:  - 2 → tumore benigno
                    - 4 → tumore maligno
            
Obiettivo del preprocessing:
Lo scopo del preprocessing è trasformare il dataset grezzo in un insieme di dati:
- numerico
- privo di valori mancanti nelle feature
- coerente con i range specificati
- contenente solo classi valide
- Il dataset preprocessato è destinato alle successive fasi di addestramento e valutazione dei modelli di machine learning.

Operazioni di preprocessing eseguite:

1. Selezione delle colonne rilevanti: vengono mantenute solo le colonne corrispondenti all’ID, alle 9 feature e alla classe.
Tutte le altre colonne vengono eliminate.
2. Rimozione delle osservazioni incomplete: vengono eliminate le righe prive di:
- identificativo del campione
- etichetta di classe
3. Pulizia e conversione dei valori numerici: sostituzione della virgola con il punto come separatore decimale e conversione della feature in formato numerico.
4. Gestione dei valori mancanti: i valori mancanti nelle feature vengono sostituiti con la mediana della colonna.
5. Pulizia delle classi:
Vengono mantenute esclusivamente le osservazioni con classe:

2 (benigno)
4 (maligno)

6. Controllo dei range:
Le osservazioni che contengono valori delle feature fuori dall’intervallo [1, 10] vengono eliminate.

Il preprocessing produce un dataset finale con le seguenti proprietà:
- feature esclusivamente numeriche
- assenza di valori NaN nelle feature
- classi limitate a 2 e 4
- valori coerenti con i range definiti dalla traccia

Stato del progetto

Attualmente il progetto include:

- preprocessing completo del dataset e suddivisione dei dati in:

X: matrice delle feature (9 variabili)
y: vettore delle etichette di classe

Le fasi successive (model development e model evaluation) verranno implementate nei prossimi step.
