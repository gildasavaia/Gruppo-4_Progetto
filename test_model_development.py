import unittest
import random
from math import sqrt

from Model_Development import (
    euclidean_distance,
    get_k_nearest,
    predict_one,
    predict_batch
)

# Test sui metodi di calcolo della distanza e predizione k-NN
class TestKNNDistances(unittest.TestCase):
    """Test sul calcolo della distanza euclidea"""

    def test_euclidean_distance_simple(self): # Test semplice con punti noti
        a = [0, 0]
        b = [3, 4]
        self.assertAlmostEqual(
            euclidean_distance(a, b),
            5.0,
            msg="La distanza euclidea tra (0,0) e (3,4) deve essere 5"
        )

    def test_euclidean_distance_zero(self): # Distanza tra un punto e se stesso
        a = [1, 2, 3]
        self.assertEqual(
            euclidean_distance(a, a),
            0.0,
        )


class TestKNearestSelection(unittest.TestCase): 
    """Test sulla selezione dei k vicini più prossimi"""

    def test_get_k_nearest(self): # Test semplice con distanze note
        distances = [
            (0.5, 2),
            (2.0, 4),
            (1.0, 2),
            (3.0, 4)
        ]

        k = 2
        nearest = get_k_nearest(distances, k) 

        self.assertEqual(len(nearest), k, "Numero di vicini errato")

        # Le distanze selezionate devono essere le più piccole
        selected_distances = [d for d, _ in nearest]
        self.assertListEqual(
            sorted(selected_distances),
            [0.5, 1.0],
            "I k vicini selezionati non sono quelli corretti"
        )


class TestKNNPrediction(unittest.TestCase):
    """Test sulla predizione k-NN con dataset artificiali"""

    def setUp(self):
        # Dataset artificiale semplice e deterministico
        self.X_train = [
            [0, 0],
            [1, 1],
            [10, 10]
        ]
        self.y_train = [2, 2, 4]

        self.rng = random.Random(0)

    def test_predict_one_majority_class(self): #Test predizione singola
        x_test = [0.5, 0.5]
        k = 2

        pred = predict_one( 
            self.X_train,
            self.y_train,
            x_test,
            k,
            self.rng
        )

        self.assertEqual( # La predizione deve essere la classe di maggioranza
            pred,
            2,
            "La predizione k-NN non restituisce la classe di maggioranza"
        )

    def test_predict_batch(self): # Test predizione batch
        X_test = [
            [0.2, 0.2],
            [9.5, 9.5]
        ]
        k = 1

        preds = predict_batch(
            self.X_train,
            self.y_train,
            X_test,
            k,
            self.rng
        )

        self.assertEqual(
            preds,
            [2, 4],
            "Le predizioni batch non sono corrette"
        )


if __name__ == "__main__":
    unittest.main()
