import unittest
import pandas as pd
from data_preprocessing import get_features_and_labels


class TestDataPreprocessing(unittest.TestCase): #test case per la funzione di preprocessing

    @classmethod
    def setUpClass(cls):
        df = pd.DataFrame({ #creazione di un DataFrame di esempio
            "Sample code number": [1001, 1002, 1003, 1004, 1005], 

            # 9 FEATURE 
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0], 
            "f2": [5.1, 4.1, 3.1, 2.1, 1.1],
            "f3": [10, 11, 12, 13, 14],
            "f4": [7, 8, 9, 6, 5],
            "f5": [0.5, 0.6, 0.7, 0.8, 0.9],
            "f6": [100, 101, 102, 103, 104],
            "f7": [1, 0, 1, 0, 1],
            "f8": [9.9, 8.8, 7.7, 6.6, 5.5],
            "f9": [3, 3, 3, 3, 3],

            # LABEL
            "classtype_v1": [2, 4, 2, 4, 2]
        })

        cls.X, cls.y = get_features_and_labels(df) #estrazione di feature e label

    def test_dataset_not_empty(self): #controlla che il dataset non sia vuoto
        self.assertGreater(len(self.X), 0, "Il dataset è vuoto")

    def test_same_number_of_samples(self): #controlla che il numero di campioni in X e y sia lo stesso
        self.assertEqual( #
            len(self.X), 
            len(self.y),
            "Features e label hanno dimensioni diverse"
        )

    def test_number_of_features(self): #controlla che il numero di feature sia 9 
        # X è list[list]
        self.assertEqual(
            len(self.X[0]),
            9,
            "Il numero di feature non è 9"
        )

    def test_class_labels_valid(self): #controlla che le classi siano solo 2 o 4
        valid_classes = {2, 4}
        self.assertTrue(
            set(self.y).issubset(valid_classes),
            "Sono presenti classi non valide"
        )

    def test_no_missing_values(self):
        # controlla None o NaN nelle liste
        for row in self.X:
            for value in row:
                self.assertIsNotNone(value, "Valore None trovato nelle feature")
