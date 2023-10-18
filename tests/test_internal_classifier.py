import unittest
from unittest.mock import patch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from structured_segmentation.learner.internal_classifier import InternalClassifier


class TestInternalClassifier(unittest.TestCase):
    def setUp(self):
        # Create an instance of InternalClassifier
        self.clf = InternalClassifier(opt={"type": "lr"})

        # Load a sample dataset for testing
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def test_fit_predict(self):
        # Test fitting the classifier and making predictions
        with patch('builtins.print'):
            self.clf.new()
            self.clf.fit(self.X_train, self.y_train)
            y_pred = self.clf.predict(self.X_test)
            self.assertEqual(y_pred.shape[0], self.X_test.shape[0])

    def test_predict_proba(self):
        # Test predict_proba method
        with patch('builtins.print'):
            self.clf.new()
            self.clf.fit(self.X_train, self.y_train)
            y_prob = self.clf.predict_proba(self.X_test)
            self.assertTrue(isinstance(y_prob, np.ndarray))
            self.assertEqual(y_prob.shape[0], self.X_test.shape[0])

    def test_evaluate(self):
        # Test the evaluate method
        with patch('builtins.print'):
            self.clf.new()
            self.clf.fit(self.X_train, self.y_train)
            f1_score = self.clf.evaluate(self.X_test, self.y_test, verbose=False)
            self.assertTrue(isinstance(f1_score, float))


if __name__ == '__main__':
    unittest.main()
