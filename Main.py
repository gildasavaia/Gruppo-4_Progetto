import random
import pandas as pd
from Model_Evaluation import holdout, random_subsampling, print_metrics, plot_confusion_matrix, save_metrics_to_excel, \
    plot_roc_curve, stratified_cv
from data_preprocessing import prepocessing, get_features_and_labels
import os

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_path, "Data")
    results_dir = os.path.join(base_path, "Results")

    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "version_1.csv")
    cm_path = os.path.join(results_dir, "confusion_matrix_knn.png")
    roc_path = os.path.join(results_dir, "roc_curve_knn.png")
    excel_path = os.path.join(results_dir, "knn_results.xlsx")
    data = pd.read_csv(csv_path)

    df=prepocessing(data)
    rng = random.Random(42)
    K_NEIGHBORS = 5
    positive_label = 4
    X, y = get_features_and_labels(df)
    data_xy = list(zip(X, y))

    print("Seleziona il metodo di validazione:")
    print("H -> Holdout")
    print("B -> Random Subsampling")
    print("C -> Stratified Cross Validation")

    choice = input("Inserisci la tua scelta (H/B/C): ").strip().upper()

    if choice == "H":

        train_perc = float(
            input("Inserisci la percentuale di dati per il training set (es. 80): ")
        )

        test_size = 1 - train_perc / 100
        metrics, cm, y_test, y_pred = holdout(
            data_xy,
            test_size,
            K_NEIGHBORS,
            rng,
            positive_label
        )
        print_metrics(metrics)
        plot_confusion_matrix(cm,cm_path)
        save_metrics_to_excel( metrics,excel_path)

        plot_roc_curve(
            y_test,
            y_pred,
            positive_label,
            roc_path
        )
    elif choice == "B":

        train_perc = float(
            input("Inserisci la percentuale di dati per il training set (es. 80): ")
        )
        test_size = 1 - train_perc / 100
        n_exp = int(input("Numero esperimenti random subsampling: "))
        metrics, cm, y_test, y_pred = random_subsampling(
            data_xy,
            test_size,
            K_NEIGHBORS,
            n_exp,
            rng,
            positive_label
        )
        print_metrics(metrics)
        plot_confusion_matrix(cm,cm_path)
        save_metrics_to_excel(metrics,excel_path )

        plot_roc_curve(
            y_test,
            y_pred,
            positive_label,
            roc_path
        )
    elif choice == "C":

        n_folds = int(input("Numero di fold per Stratified CV: "))

        metrics, cm, y_test, y_pred = stratified_cv(
            data_xy,
            n_folds,
            K_NEIGHBORS,
            rng,
            positive_label
        )
        print_metrics(metrics)
        plot_confusion_matrix(cm, cm_path)
        save_metrics_to_excel(metrics,excel_path)

        plot_roc_curve(
            y_test,
            y_pred,
            positive_label,
            roc_path
        )

    else:
        raise ValueError("Scelta non valida. Inserire H, B o C.")
    
if __name__ == "__main__":
    main()