import unittest
import numpy as np
from numcompute.optim import grad, jacobian


class TestGrad(unittest.TestCase):
    def test_grad_simple_quadratic(self):
        # f(x) = x1^2 + x2^2 → grad = [2x1, 2x2]
        def f(x):
            return x[0]**2 + x[1]**2

        x = np.array([3.0, 4.0])
        result = grad(f, x)

        expected = np.array([6.0, 8.0])
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_grad_linear(self):
        # f(x) = 3x1 + 5x2 → grad = [3,5]
        def f(x):
            return 3*x[0] + 5*x[1]

        x = np.array([10.0, -2.0])
        result = grad(f, x)

        expected = np.array([3.0, 5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_grad_invalid_shape(self):
        def f(x):
            return x**2

        x = np.array([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            grad(f, x)

    def test_grad_method_error(self):
        def f(x):
            return x[0]**2

        x = np.array([1.0])
        with self.assertRaises(ValueError):
            grad(f, x, method="invalid")

class TestJacobian(unittest.TestCase):
    def test_jacobian_simple(self):
        # F(x) = [x1^2, x2^2]
        def F(x):
            return np.array([x[0]**2, x[1]**2])

        x = np.array([3.0, 4.0])
        result = jacobian(F, x)

        expected = np.array([
            [6.0, 0.0],
            [0.0, 8.0]
        ])

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_jacobian_mixed(self):
        # F(x) = [x1 + x2, x1 * x2]
        def F(x):
            return np.array([x[0] + x[1], x[0] * x[1]])

        x = np.array([2.0, 3.0])
        result = jacobian(F, x)

        expected = np.array([
            [1.0, 1.0],
            [3.0, 2.0]
        ])

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_jacobian_invalid_input(self):
        def F(x):
            return np.array([x[0]])

        x = np.array([[1.0]])
        with self.assertRaises(ValueError):
            jacobian(F, x)

    def test_jacobian_invalid_output(self):
        def F(x):
            return x[0]**2  # not vector

        x = np.array([1.0])
        with self.assertRaises(ValueError):
            jacobian(F, x)


if __name__ == "__main__":
    unittest.main()