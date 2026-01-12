import argparse
import os
import random
import pandas as pd
# Funzioni per valutazione del modello
from Model_Evaluation import (
    holdout,
    random_subsampling,
    stratified_cv,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_to_excel
)

# Funzioni di preprocessing
from data_preprocessing import prepocessing, get_features_and_labels


def parse_arguments():
    """Definisce e legge gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline di classificazione KNN con diversi metodi di validazione"
    )

    #Parametri principali
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Percorso al file CSV di input"
    )
    # Cartella di output dei risultati
    parser.add_argument(
        "--results-dir",
        type=str,
        default="Results",
        help="Cartella dove salvare i risultati"
    )
    # Numero di vicini per KNN
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Numero di vicini (KNN)"
    )

    parser.add_argument(
        "--positive-label",
        type=int,
        default=4,
        help="Etichetta considerata positiva"
    )

    # Metodo di validazione
    parser.add_argument(
        "--validation",
        type=str,
        choices=["holdout", "subsampling", "cv"],
        required=True,
        help="Metodo di validazione: holdout | subsampling | cv"
    )
    # Percentuale di dati per il training set
    parser.add_argument(
        "--train-perc",
        type=float,
        default=80.0,
        help="Percentuale di dati per il training (holdout/subsampling)"
    )
    # Numero di esperimenti per il random subsampling
    parser.add_argument(
        "--n-exp",
        type=int,
        default=10,
        help="Numero di esperimenti per random subsampling"
    )
    # Numero di fold per la cross validation
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Numero di fold per Stratified Cross Validation"
    )
    # Seed per rendere i risultati riproducibili
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità"
    )

    return parser.parse_args()


def main():
    # Lettura degli argomenti da CLI
    args = parse_arguments()
    # Inizializzazione del generatore casuale

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Data path
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File CSV non trovato: {data_path}")

    # Results directory
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BASE_DIR, results_dir)


    rng = random.Random(args.seed)

    os.makedirs(results_dir, exist_ok=True)

    #  Percorsi output
    cm_path = os.path.join(results_dir, "confusion_matrix_knn.png")
    roc_path = os.path.join(results_dir, "roc_curve_knn.png")
    excel_path = os.path.join(results_dir, "knn_results.xlsx")
    processed_path = os.path.join(results_dir, "data_processed.csv")

    # Caricamento dati
    data = pd.read_csv(data_path)
    # Preprocessing dei dati
    df = prepocessing(data)
    df.to_csv(processed_path, index=False)
    # Estrazione feature e label
    X, y = get_features_and_labels(df)
    data_xy = list(zip(X, y))

    # Selezione del metodo di validazione
    if args.validation == "holdout":
        test_size = 1 - args.train_perc / 100

        metrics, cm, y_test, y_pred = holdout(
            data_xy,
            test_size,
            args.k,
            rng,
            args.positive_label
        )

    elif args.validation == "subsampling":
        test_size = 1 - args.train_perc / 100

        metrics, cm, y_test, y_pred = random_subsampling(
            data_xy,
            test_size,
            args.k,
            args.n_exp,
            rng,
            args.positive_label
        )

    elif args.validation == "cv":
        metrics, cm, y_test, y_pred = stratified_cv(
            data_xy,
            args.n_folds,
            args.k,
            rng,
            args.positive_label
        )

    else:
        raise ValueError("Metodo di validazione non valido")

    # Output
    print_metrics(metrics)
    plot_confusion_matrix(cm, cm_path)
    plot_roc_curve(y_test, y_pred, args.positive_label, roc_path)
    save_metrics_to_excel(metrics, excel_path)

# Avvio del programma
if __name__ == "__main__":
    main()
