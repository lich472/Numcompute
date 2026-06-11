import unittest
import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.metrics import StreamingMetrics
from numcompute_stream.stream import StreamTrainer


class DummyEstimator:
    def fit(self, X, y):
        self.majority_ = int(np.bincount(y.astype(int)).argmax())
        self.fitted = True
        return self

    def partial_fit(self, X, y):
        self.majority_ = int(np.bincount(y.astype(int)).argmax())
        self.fitted = True
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.majority_)


class TestStreamTrainer(unittest.TestCase):

    def setUp(self):
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('model', DummyEstimator())
        ])
        metrics = StreamingMetrics()
        self.trainer = StreamTrainer(pipe, metrics)

    def test_fit_and_score_flow(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        self.trainer.fit_chunk(X, y)
        self.assertEqual(self.trainer.chunk_count, 1)
        self.assertEqual(len(self.trainer.log), 1)
        self.assertEqual(self.trainer.log[0]['type'], 'fit')
        
        results = self.trainer.score_chunk(X, y)
        self.assertIn('accuracy', results)
        self.assertEqual(len(self.trainer.log), 2)

    def test_multiple_chunks(self):

        X1 = np.array([[1, 2], [3, 4]])
        y1 = np.array([0, 1])
        X2 = np.array([[5, 6], [7, 8]])
        y2 = np.array([0, 1])
        
        self.trainer.fit_chunk(X1, y1)
        self.trainer.fit_chunk(X2, y2)
        
        self.assertEqual(self.trainer.chunk_count, 2)
        self.assertIsInstance(self.trainer.get_log(), list)

    def test_reset_functionality(self):

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        self.trainer.fit_chunk(X, y)
        self.trainer.reset()
        
        self.assertEqual(self.trainer.chunk_count, 0)
        self.assertEqual(len(self.trainer.log), 0)


if __name__ == "__main__":
    unittest.main()