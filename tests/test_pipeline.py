import unittest
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler


class DummyEstimator:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_class_ = self.classes_[0]
        self.fitted = True
        return self

    def predict(self, X):
        if not getattr(self, "fitted", False):
            raise RuntimeError("DummyEstimator not fitted.")
        return np.full(X.shape[0], self.majority_class_)
    
    def partial_fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_class_ = self.classes_[0]
        self.fitted = True
        return self


class SimpleTransformer:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) + 1


class TestPipelineValidation(unittest.TestCase):
    def test_empty_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_duplicate_step_names(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("scale", StandardScaler()),
                ("scale", MinMaxScaler()),
            ])

    def test_invalid_step_format(self):
        with self.assertRaises(ValueError):
            Pipeline([StandardScaler()])

    def test_missing_transformer_methods(self):
        with self.assertRaises(TypeError):
            Pipeline([
                ("bad", object()),
                ("scale", StandardScaler()),
            ])


class TestPipelineTransformerOnly(unittest.TestCase):
    def test_fit_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("minmax", MinMaxScaler()),
        ])

        result = pipe.fit_transform(X)

        self.assertEqual(result.shape, X.shape)
        self.assertTrue(pipe.fitted)

    def test_transform_before_fit(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        with self.assertRaises(RuntimeError):
            pipe.transform(np.array([[1.0, 2.0]]))

    def test_predict_on_transformer_pipeline_raises(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        pipe.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        with self.assertRaises(TypeError):
            pipe.predict(np.array([[1.0, 2.0]]))

    def test_fallback_fit_transform(self):
        X = np.array([[1, 2], [3, 4]])

        pipe = Pipeline([
            ("simple", SimpleTransformer()),
            ("scale", StandardScaler()),
        ])

        result = pipe.fit_transform(X)
        self.assertEqual(result.shape, X.shape)


class TestPipelineEstimator(unittest.TestCase):
    def test_estimator_pipeline_fit_predict(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])

        pipe.fit(X, y)
        pred = pipe.predict(X)

        self.assertEqual(pred.shape, y.shape)

    def test_estimator_requires_y(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])

        with self.assertRaises(ValueError):
            pipe.fit(X)

    def test_fit_transform_with_estimator_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])

        with self.assertRaises(TypeError):
            pipe.fit_transform(X, y)


class TestPipelineParams(unittest.TestCase):
    def test_getitem(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        self.assertIsInstance(pipe["scale"], StandardScaler)

    def test_set_params(self):
        pipe = Pipeline([
            ("scale", StandardScaler(copy=True)),
        ])

        pipe.set_params(scale__copy=False)
        self.assertFalse(pipe["scale"].copy)

    def test_set_params_invalid_format(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        with self.assertRaises(ValueError):
            pipe.set_params(copy=False)

    def test_set_params_invalid_step(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        with self.assertRaises(ValueError):
            pipe.set_params(model__copy=False)

class DummyEstimatorNoPartialFit:
    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

class TestPipelinePartialFit(unittest.TestCase):
    def test_partial_fit_basic(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])
        pipe.partial_fit(X, y)
        self.assertTrue(pipe.fitted)

    def test_partial_fit_multiple_chunks(self):
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        y1 = np.array([0, 1])
        y2 = np.array([0, 1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])
        pipe.partial_fit(X1, y1)
        pipe.partial_fit(X2, y2)
        self.assertTrue(pipe.fitted)

    def test_partial_fit_requires_y_for_estimator(self):
        X = np.array([[1, 2], [3, 4]])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])
        with self.assertRaises(ValueError):
            pipe.partial_fit(X)

    def test_partial_fit_no_partial_fit_raises(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimatorNoPartialFit()),
        ])
        with self.assertRaises(TypeError):
            pipe.partial_fit(X, y)

    def test_partial_fit_predict(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DummyEstimator()),
        ])
        pipe.partial_fit(X, y)
        pred = pipe.predict(X)
        self.assertEqual(pred.shape, y.shape)
