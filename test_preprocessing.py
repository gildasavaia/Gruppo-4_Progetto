import unittest
import data_preprocessing as dp
import pandas as pd


class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Otteniamo features e label dal preprocessing
        cls.X, cls.y = dp.get_features_and_labels()

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.X), 0, "Il dataset è vuoto")

    def test_same_number_of_samples(self):
        self.assertEqual(
            len(self.X),
            len(self.y),
            "Features e label hanno dimensioni diverse"
        )

    def test_number_of_features(self):
        self.assertEqual(
            self.X.shape[1],
            9,
            "Il numero di feature non è 9"
        )

    def test_class_labels_valid(self):
        valid_classes = {2, 4}
        self.assertTrue(
            set(self.y.unique()).issubset(valid_classes),
            "Sono presenti classi non valide"
        )

    def test_no_missing_values(self):
        self.assertFalse(
            self.X.isnull().values.any(),
            "Sono presenti valori NaN nelle feature"
        )


if __name__ == "__main__":
    unittest.main()
