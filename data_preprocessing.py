import pandas as pd

# 1. Caricamento dataset
data = pd.read_csv("version_1.csv")
# Legge il file CSV "version_1.csv" e lo carica in un DataFrame pandas chiamato "data".

# 2. Selezione colonne rilevanti
#mantiene solo le colonne elencate, riduce il dataset 
cols = [
    
        "Sample code number",
        "clump_thickness_ty",
        "uniformity_cellsize_xx",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "bareNucleix_wrong",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "classtype_v1"
    ]
data = data[cols]

#3.rimozionne righe senza ID o classe
data.dropna(
    subset=["Sample code number", "classtype_v1"],
    inplace = True
)

# 4. Pulizia valori numerici (converte da virgola a punto)
string_cols = data.select_dtypes(include="object").columns

data[string_cols] = (
    data[string_cols]
    .replace(",", ".", regex=True)
)

#5 conversioe delle colonne a numerico
feature_cols = data.columns.drop(
    ["Sample code number", "classtype_v1"]
)

data[feature_cols] = data[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)

data["classtype_v1"] = pd.to_numeric(
    data["classtype_v1"], errors="raise"
)

# 6. sostituisce i NaN nelle feature con la mediana delle colonne
for col in feature_cols:
    data[col] = data[col].fillna(data[col].median())


# 7. Pulizia classi per mantenere solo i valori 2 e 4
data = data[data["classtype_v1"].isin([2, 4])]

#8. controllo range feature, elimina le righe che contengono i valori fuori dal range [1,10]
data = data[
    (data[feature_cols] >= 1).all(axis=1) &
    (data[feature_cols] <= 10).all(axis=1)
]

data.to_csv("dataset_preprocessato.csv", index=False) #salvataggio dataset


# Divisione del dataset

X = data.drop(columns=["Sample code number", "classtype_v1"]) #matrice delle feature

y = data["classtype_v1"] #class label, vettore delle etichette di classe

# Controlli di verifica

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nDistribuzione classi:")
print(y.value_counts())

"""

Un test di correttezza sensato per il tuo preprocessing deve controllare almeno che:

- il dataset non sia vuoto
- X e y abbiano lo stesso numero di righe
- X abbia esattamente 9 feature
- y contenga solo classi valide (2 e 4)
- non ci siano valori Na

"""

def get_features_and_labels(df):
    #Restituisce X (features) e y (class label)
    X = df.drop(columns=["Sample code number", "classtype_v1"])
    y = df["classtype_v1"]

    X = X.values.tolist()
    y = y.values.tolist()
    return X, y